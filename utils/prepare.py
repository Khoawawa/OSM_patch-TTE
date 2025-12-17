import json
import os
import pickle

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import SmoothL1Loss, MSELoss
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from utils.util import StandardScaler2
from PIL import Image
import torchvision.transforms as T
from models.OSM_BE_Resnet_TTE import OSM_BER_TTE, Regional_TTE
from rtree import index
from scipy.spatial import KDTree


highway = {'living_street':1, 'morotway':2, 'motorway_link':3, 'plannned':4, 'trunk':5, "secondary":6, "trunk_link":7, "tertiary_link":8, "primary":9, "residential":10, "primary_link":11, "unclassified":12, "tertiary":13, "secondary_link":14}
node_type = {'turning_circle':1, 'traffic_signals':2, 'crossing':3, 'motorway_junction':4, "mini_roundabout":5}
class RegionEmbeddingManager:
    def __init__(self, region_json, args):
        """
        region_json: list of region dicts
        args.absPath: base path for embeddings
        """

        centres = []
        features = []

        for r in region_json:
            centres.append([
                r["center"]["x"],
                r["center"]["y"],
            ])

            emb_path = os.path.join(args.absPath, r["embedding_path"])
            features.append(torch.load(emb_path, map_location=torch.device('cpu')))  # [F], CPU

        # [R, 2] region centers (CPU)
        self.centres = torch.tensor(centres, dtype=torch.float32)

        # [R, F] region embeddings (CPU)
        self.features = torch.stack(features, dim=0)

        # KD-tree built on centers (CPU)
        self.kdtree : KDTree = KDTree(self.centres.numpy())

        print(f"KDTree initialized with {len(self.centres)} regions")

    @torch.no_grad()
    def find_n_nearest_region(self, xs, ys, k):
        """
        xs, ys: 1D arrays or tensors of length N (CPU)
        k: number of nearest regions

        returns:
            centres  -> [N, k, 2]
            features -> [N, k, F]
        """

        if torch.is_tensor(xs):
            xs = xs.cpu().numpy()
        if torch.is_tensor(ys):
            ys = ys.cpu().numpy()

        query = np.stack([xs, ys], axis=1)  # [N, 2]

        _, idx = self.kdtree.query(query, k=k)  # [N, k]

        idx = torch.from_numpy(idx).long()

        centres = self.centres[idx]    # [N, k, 2]
        features = self.features[idx]  # [N, k, F]

        return centres, features
    
def collate_func(data, args, info_all):

    region_manager, edgeinfo, nodeinfo, scaler, scaler2 = info_all

    time = torch.Tensor([d[-1] for d in data])
    linkids = [np.asarray(d[1]) for d in data]
    dateinfo = [d[2:5] for d in data]
    inds = [d[0] for d in data]
    lens = np.asarray([len(k) for k in linkids], dtype=np.int16)
    
    def info(xs, date):
        infos = []
        length = 0
        for x in xs:
            info = edgeinfo[x]
            infot = []
            infot.append(highway[info[0]] if info[0] in highway.keys() else 0)
            infot.append(info[1])
            infot.append(length)
            length += info[1]
            infot += list(date)
            try:
                infot += [nodeinfo[info[2]][0],nodeinfo[info[2]][1],nodeinfo[info[3]][0],nodeinfo[info[3]][1]]
            except:
                print(info)
            infos.append(np.asarray(infot))
            # highway length sumoflength date3 gps4

        return infos

    con_links = np.concatenate([info(b, dateinfo[ind]) for ind, b in enumerate(linkids)], dtype='object')
    gps = con_links[:, 6:8].astype(np.float32).reshape(-1, 2)
    
    region_center, region_feature = region_manager.find_n_nearest_region(gps[:,0], gps[:,1], 1)
    region_center = region_center.squeeze(1)
    gps = torch.from_numpy(gps).float()
    offset = (gps - region_center) / args.data_config['patch']['patch_size']  # shape: [total_links, 2]
    # print(region_feature.shape)

    mask = np.arange(lens.max()) < lens[:, None] # mask.shape = [batch_size, max_len]

    padded = np.zeros((*mask.shape, 1+2+3+4), dtype=np.float32)
    
    con_links[:, 1:3] = scaler.transform(con_links[:, 1:3])
    con_links[:, 6:10] = scaler2.transform(con_links[:, 6:10])

    padded[mask] = con_links
    rawlinks = np.full(mask.shape, fill_value=args.data_config['edges'] + 1, dtype=np.int16)
    rawlinks[mask] = np.concatenate(linkids)

    def random_mask(tokens: np.array, rate: float):
        replaces = np.where(np.random.random(len(tokens)) <= rate)[0]
        labels = np.full(len(tokens),dtype=np.int16, fill_value=-100)
        tokens = tokens.copy()

        labels[replaces] = tokens[replaces]
        tokens[replaces] = np.asarray([args.data_config['edges'] + 1] * len(replaces))   # 此处直接赋值会改变dataset原始值，应该考虑采用深拷贝复制一份新数组再更改
        return labels, tokens


    mask_label_tmp = []
    sub_input_tmp = []
    for k in linkids:
        tmp1, tmp2 = random_mask(k, rate=args.mask_rate)
        mask_label_tmp.append(tmp1)
        sub_input_tmp.append(tmp2)
    mask_label = np.full(mask.shape, dtype=np.int16, fill_value=-100)
    mask_label[mask] = np.concatenate(mask_label_tmp)

    linkindex = np.full(mask.shape, fill_value=args.data_config['edges'] + 1, dtype=np.int16)
    linkindex[mask] = np.concatenate(sub_input_tmp)
    mask_encoder = np.zeros(mask.shape, dtype=np.int16)
    mask_encoder[mask] = np.concatenate([[1]*k for k in lens])

    return {'links':torch.from_numpy(padded),
            'region_feature': region_feature,
            'offset': offset,
            'valid_mask': mask,
            'lens':torch.LongTensor(lens), 
            'inds': inds, 
            'mask_label': torch.LongTensor(mask_label),
            "linkindex":torch.LongTensor(linkindex), 
            'rawlinks': torch.LongTensor(rawlinks),
            'encoder_attention_mask': torch.LongTensor(mask_encoder)
            }, time

class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size
        if isinstance(dataset[0], dict):
            self.lengths = [len(d['lats']) for d in dataset]
        elif isinstance(dataset[0][1], list):
            self.lengths = [len(d[1]) for d in dataset]
        else:
            self.lengths = [d[0]['lens'] for d in dataset]
        self.indices = list(range(self.count))

    def __iter__(self):
        '''
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        '''
        np.random.shuffle(self.indices)

        chunk_size = self.batch_size * 100

        chunks = (self.count + chunk_size - 1) // chunk_size

        # re-arrange indices to minimize the padding
        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size: (i + 1) * chunk_size]
            partial_indices.sort(key = lambda x: self.lengths[x], reverse = True)
            self.indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        # yield batcha
        batches = (self.count - 1 + self.batch_size) // self.batch_size

        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size

def load_datadoct_pre(args):
    global info_all
    abspath = os.path.join(os.path.dirname(__file__), "data_config.json")
    
    with open(abspath) as file:
        data_config = json.load(file)[args.dataset]
        args.data_config = data_config
    
    with open(os.path.join(args.absPath,args.data_config['edges_dir']), 'rb') as f:
        edgeinfo = pickle.load(f)
    with open(os.path.join(args.absPath,args.data_config['nodes_dir']), 'rb') as f:
        nodeinfo = pickle.load(f)
        
    with open(os.path.join(args.absPath,args.data_config['patch']['patch_json']), 'r') as f:
        patch_json = json.load(f)
        
    region_manager = RegionEmbeddingManager(patch_json, args)

    if "porto" in args.dataset:
        scaler = StandardScaler()
        scaler.fit([[0, 0]])
        scaler.mean_ = [107.497195, 3010.37456]
        scaler.scale_ = [131.102877, 2750.78118]
        scaler2 = StandardScaler()
        scaler2.fit([[0, 0, 0, 0]])
        scaler2.mean_ = [-8.62247695, 41.15923239, -8.62256569, 41.15929004]
        scaler2.scale_ = [0.02520552, 0.01236445, 0.02526226, 0.01242564]

        
    elif "chengdu" in args.dataset:
        scaler = StandardScaler()
        scaler.fit([[0,0]])
        scaler.mean_ = [188.285260, 3969.52982]
        scaler.scale_ = [206.040346, 3658.76429]
        scaler2 = StandardScaler()
        scaler2.fit([[0,0,0,0]])
        scaler2.mean_ = [104.06379941,  30.65844312, 104.06381633,  30.65845601]
        scaler2.scale_ = [0.03480474, 0.02717924, 0.03484908, 0.02719959]
    else:
        ValueError("Wrong Dataset Name")

    info_all = [region_manager,edgeinfo, nodeinfo, scaler, scaler2]
    

class Datadict(Dataset):
    def __init__(self, inputs):
        self.content = inputs

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(self.content)
def load_test_datadict(args):
    tdata = np.load(os.path.join(args.absPath,args.data_config['data_dir'],'test.npy'), allow_pickle=True)
    test_loader = DataLoader(Datadict(tdata), batch_sampler=BatchSampler(tdata, args.data_config['batch_size']),
                                        collate_fn=lambda x: collate_func(x, args, info_all),
                                        pin_memory=True)
    
    return test_loader, StandardScaler2(mean=args.data_config['time_mean'], std=args.data_config['time_std'])
def load_datadict(args):
    data = {}
    loader = {}
    if args.mode == 'test':
        phases = ['test']
    else:
        phases = ['train', 'val']

    for phase in phases:
        tdata = np.load(os.path.join(args.absPath,args.data_config['data_dir'], phase + '.npy'), allow_pickle=True)
        data[phase] = tdata

        if phase == 'train':
            loader[phase] = DataLoader(Datadict(data[phase]), batch_sampler=BatchSampler(data[phase], args.data_config['batch_size']),
                                        collate_fn=lambda x: collate_func(x, args, info_all),
                                        pin_memory=True)
        else:
            
            loader[phase] = DataLoader(Datadict(data[phase]), batch_size=args.data_config['batch_size'],
                                        collate_fn=lambda x: collate_func(x, args, info_all),
                                        shuffle=False, pin_memory=True)
    return loader.copy(), StandardScaler2(mean=args.data_config['time_mean'], std=args.data_config['time_std'])


def create_model(args):
    absPath = os.path.join(os.path.dirname(__file__), "model_config.json")
    with open(absPath) as file:
        model_config = json.load(file)[args.model]
    args.model_config = model_config
    model_config['pad_token_id'] = args.data_config['edges'] + 1
    if "OSM_BER_TTE" in args.model:
        return OSM_BER_TTE(**model_config)
    if "region" in args.model.lower():
        return Regional_TTE(**model_config)

        

def create_main_loss(loss_bert,loss, args):
    beta = args.beta
    bert_weight  = 1 - beta
        
    return bert_weight*loss_bert / (loss_bert / loss + 1e-4).detach()\
            + beta * loss\

def create_loss(args):
    if args.loss == 'rmse':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            rmse = torch.sqrt(torch.mean(torch.pow(preds - labels, 2)))
            return rmse
    elif args.loss == 'mse':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            # mse = torch.mean(torch.pow(preds - labels, 2))
            mse = MSELoss(reduction='mean').forward(preds.view(-1), labels)
            return mse
    elif args.loss == 'mape':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            mape = torch.mean(torch.abs(preds - labels) / (labels + 0.1))
            return mape
    elif args.loss == 'mae':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            mape = torch.mean(torch.abs(preds - labels))
            return mape
    elif args.loss == 'smoothL1':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            preds = torch.squeeze(preds, 1)
            smoothL1 = SmoothL1Loss(reduction='mean', beta = args.loss_val).forward(preds, labels)
            return smoothL1

    else:
        raise ValueError("Unknown loss function.")
    return loss


