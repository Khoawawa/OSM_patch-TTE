import os
import sys
import shutil
from tqdm import tqdm

import torch
from torch import optim
import numpy as np

from train.train_model import train_model
from utils.prepare import create_model, create_loss
from utils.prepare import load_datadict, load_datadoct_pre
from utils.prepare import load_test_datadict
from utils.metric import calculate_metrics
from utils.util import to_var
import time

@torch.no_grad()
def test_model(model, data_loader, args):
    model.eval()
    predictions = list()
    targets = list()
    inds = list()
    tqdm_loader = tqdm(data_loader)
    for features, truth_data in tqdm_loader:
        if isinstance(features, dict) and 'inds' in features.keys():    
            inds.append(features['inds'])
        features = to_var(features, args.device)
        truth_data = to_var(truth_data, args.device)

        with torch.no_grad():
            with torch.amp.autocast(args.device):
                outputs, _ = model(features, args,'test')                        

        targets.append(truth_data.cpu().numpy())
        predictions.append(outputs.cpu().detach().numpy())

    pre2 = np.concatenate(predictions).squeeze()
    tar2 = np.concatenate(targets)
    if len(inds) > 0:
        print(f"test size: {len(inds)}")
        print(f"test traj ids of a batch: {inds[0]}")
        inds = np.concatenate(inds)
    else:
        inds = None
    metric = calculate_metrics(pre2, tar2, args, plot=True, inds=inds)
    print(metric)
    with open(f'{args.absPath}/data/result_{args.model}.txt', 'a') as f:
        f.write(time.strftime("%m/%d %H:%M:%S",time.localtime(time.time())))
        f.write(f"epoch:{args.epochs} lr:{args.lr}\ndataset:{args.dataset} identify:{args.identify}\nloss:{args.loss}\n")
        f.write(f"{args.model_config}\n")
        f.write(f"{args.data_config}\n")
        f.write(f"{metric}\n\n")

    np.save(os.path.join(args.model_folder, "result.npy"), np.asarray([pre2, inds]))


def test_main(args):
    if args.model == 'None':
        print('No chosen model')
        sys.exit(0)
    print(f"Test {args.model}_{args.identify} on {args.dataset}")
    
    load_datadoct_pre(args)
    test_loader, scaler = load_datadict(args)
    args.scaler = scaler
    
    model = create_model(args)
    
    model_folder = f'{args.absPath}/data/save_models/{args.model}_{args.identify}_{args.dataset}'
    args.model_folder = model_folder
    model = model.to(args.device)
    
    print(f'loss function: {args.loss}')
    print(f'model config: {args.model_config}')
    print(f'data config: {args.data_config}')
    print(f'arg: {args}')
    final_model = torch.load(os.path.join(model_folder, 'final_model.pkl'), map_location=args.device)
    model.load_state_dict(final_model['state_dict'], strict=False)
    test_model(model, test_loader, args)    
    