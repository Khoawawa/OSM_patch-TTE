import os
import sys
import shutil
import json
from tqdm import tqdm

import torch
import numpy as np

from utils.prepare import create_model
from utils.prepare import  load_datadoct_pre
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
    log = dict()
    for step, (features, truth_data) in enumerate(tqdm_loader):
        if isinstance(features, dict) and 'inds' in features.keys():    
            inds.append(features['inds'])
        features = to_var(features, args.device)
        truth_data = to_var(truth_data, args.device)

        with torch.amp.autocast(args.device):
            outputs, log_batch = model(features, args,is_log=True)                        
        for k in log_batch.keys():
            if k not in log:
                log[k] = []
            log[k].append(log_batch[k].cpu().numpy())
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
    json_log = {
        k: np.concatenate(v).astype(float).tolist()
        for k, v in log.items()
    }
    with open(f'{args.absPath}/data/log_{args.model}.json', 'w') as f:
        json.dump(json_log, f, indent=4)
        f.write('\n')
    


def test_main(args):
    if args.model == 'None':
        print('No chosen model')
        sys.exit(0)
    print(f"Test {args.model}_{args.identify} on {args.dataset}")
    
    load_datadoct_pre(args)
    test_loader, scaler = load_test_datadict(args)
    args.scaler = scaler
    
    model = create_model(args)
    
    model_folder = f'{args.absPath}/data/save_models/{args.model}_{args.identify}_{args.dataset}'
    args.model_folder = model_folder
    model = model.to(args.device)

    print(f'model config: {args.model_config}')
    print(f'data config: {args.data_config}')
    print(f'arg: {args}')
    best_model = torch.load(os.path.join(model_folder, 'best_model.pkl'), map_location=args.device)
    model.load_state_dict(best_model['state_dict'], strict=False)
    test_model(model, test_loader, args)    
    