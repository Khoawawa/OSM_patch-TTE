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
from models.POI_MulT_TTE import POI_MulT_TTE

highway = {'living_street':1, 'morotway':2, 'motorway_link':3, 'plannned':4, 'trunk':5, "secondary":6, "trunk_link":7, "tertiary_link":8, "primary":9, "residential":10, "primary_link":11, "unclassified":12, "tertiary":13, "secondary_link":14}
node_type = {'turning_circle':1, 'traffic_signals':2, 'crossing':3, 'motorway_junction':4, "mini_roundabout":5}
poi_type = {'education':0,'business':1,'religious':2,'commercial':3,'transport_hub':4,'event_venue':5,'none':6}
def collate_func(data, args, info_all):
    edgeinfo, nodeinfo, scaler, scaler2, regions, global_density = info_all

    time = torch.Tensor([d[-1] for d in data])
    linkids = []
    dateinfo = []
    inds = []
    for _, l in enumerate(data):
        linkids.append(np.asarray(l[1]))
        dateinfo.append(l[2:5])
        inds.append(l[0])
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
    
    # need to find the 7x7 cells
    lon,lat = con_links[:, 6], con_links[:, 7] # start lat lon, (n,)
    cell_size = args.data_config['cell_size']
    m = args.data_config['m']
    min_lat, min_lon = args.data_config['min_lat'], args.data_config['min_lon']
    # find out which cell the segment belong to
    cell_lons = np.floor((lon - min_lon) / cell_size).astype(np.int64) # n,
    cell_lats = np.floor((lat - min_lat) / cell_size).astype(np.int64) # n,
    
    assert cell_lons.min() >= 0
    assert cell_lats.min() >= 0
    assert cell_lons.max() < global_density.shape[0]
    assert cell_lats.max() < global_density.shape[1]
    # (n,m*m,T)
    poi_matrix = local_poi_extraction(cell_lons, cell_lats, global_density, m)
    poi_matrix = poi_matrix.float()

    mask = np.arange(lens.max()) < lens[:, None]
    mask_tensor = torch.from_numpy(mask)
    # reshape poi_matrix to sequence -> (batch, seq_len, m*m, T)
    poi_matrix_padded = torch.zeros((*mask.shape, m*m, poi_matrix.shape[2]), dtype=torch.float32)
    poi_matrix_padded[mask_tensor] = poi_matrix
    
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
            'poi_matrix': poi_matrix_padded,
            'valid_mask': mask,
            'lens':torch.LongTensor(lens), 
            'inds': inds, 
            'mask_label': torch.LongTensor(mask_label),
            "linkindex":torch.LongTensor(linkindex), 
            'rawlinks': torch.LongTensor(rawlinks),
            'encoder_attention_mask': torch.LongTensor(mask_encoder)
            }, time

def local_poi_extraction(cell_lons, cell_lats, global_density, m):
    # cell_lons, cell_lats: (n,)
    # global_density: (H+pad, W+pad, T)
    device = global_density.device
    pad = m // 2
    
    offsets = torch.arange(-pad, pad + 1) # (-2,-1,0,1,2) for m=5
    delta_lons, delta_lats = torch.meshgrid(offsets, offsets, indexing='ij') # (m,m)
    
    delta_lons = delta_lons.reshape(-1) # (m*m,)
    delta_lats = delta_lats.reshape(-1) # (m*m,)

    center_lons = torch.as_tensor(cell_lons,dtype=torch.long, device=device).unsqueeze(1) + pad # (n,1)
    center_lats = torch.as_tensor(cell_lats, dtype=torch.long, device=device).unsqueeze(1) + pad # (n,1)
    
    rows = center_lons + delta_lons # (n, m*m)
    cols = center_lats + delta_lats # (n, m*m)
    
    poi_matrix = global_density[rows, cols, :] # (n, m*m, T)
    
    return poi_matrix
    
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
    with open(os.path.join(args.data_config['poi_json']), 'r') as f:
        pois_data = json.load(f)
    # precomputing global poi density matrix
    T = len(poi_type)
    max_clon = max(v["cell_id"][0] for v in pois_data.values())
    max_clat = max(v["cell_id"][1] for v in pois_data.values())
    
    H = max_clon + 1
    W = max_clat + 1
    global_density = np.zeros((H, W, T), dtype=np.int16)
    
    for data in pois_data.values():
        clon, clat = data['cell_id']
        if not data['pois']:
            t_idx = poi_type['none']
            global_density[clon, clat, t_idx] += 1
            continue
        for poi in data['pois']:
            type_ = poi['type']
            t_idx = poi_type[type_] if type_ in poi_type.keys() else -1
            if 0 <= t_idx < T:
                global_density[clon, clat, t_idx] += 1

    m = args.data_config['m']
    pad = m // 2
    global_density_tensor = torch.from_numpy(global_density)
    padded_density = torch.nn.functional.pad(
        global_density_tensor.permute(2, 0, 1), # [T, H, W]
        (pad, pad, pad, pad), 
        mode='constant', value=0
    ).permute(1, 2, 0) # Back to [H+pad, W+pad, T]
    
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

    info_all = [edgeinfo, nodeinfo, scaler, scaler2, pois_data, padded_density]


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
        phases = ['train', 'val', 'test']

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
    model_config['poi_type_size'] = len(poi_type)
    model_config['pad_token_id'] = args.data_config['edges'] + 1
    
    return POI_MulT_TTE(**model_config)
        

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

