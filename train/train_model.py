import copy
import time
from typing import Dict
import gc

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metric import calculate_metrics
from utils.util import save_model, to_var
from utils.prepare import create_main_loss
def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag
        
def train_model(model: nn.Module, data_loaders: Dict[str, DataLoader],
                loss_func: callable, optimizer: torch.optim,
                model_folder: str, args, start_epoch=-1, **kwargs):
    num_epochs = args.epochs
    phases = [
        'train',
        'val',
        'test'
        ]
    since = time.perf_counter()
    for phase in phases:
        if phase not in data_loaders:
            raise KeyError(f"{phase} loader is missing from data_loaders")
        print(f"{phase} loader found with {len(data_loaders[phase])} batches")
        
    with open(model_folder + "/output.txt", "a") as f:
        f.write(str(model))
        f.write("\n\n")

    save_dict, best_mae = {'state_dict': copy.deepcopy(model.state_dict()),
                           'epoch': 0
                           }, 10000    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.2, patience=2,
    #                                                  threshold=1e-2, threshold_mode='rel', min_lr=1e-7)
    # if hasattr(args, 'scheduler_state_dict'):
    #     scheduler.load_state_dict(args.scheduler_state_dict)
    print("LR: ", optimizer.param_groups[0]['lr'])
    scaler = torch.amp.GradScaler()
    try:
        for epoch in range(start_epoch + 1, num_epochs):
            running_loss = {phase: 0.0 for phase in phases}
            msg = []
            # training/val/test loop
            for phase in phases:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                    
                steps, predictions, targets = 0, list(), list()
                
                tqdm_loader = tqdm(data_loaders[phase],mininterval=3)
                for step, (features, truth_data) in enumerate(tqdm_loader):
                    steps += truth_data.size(0)
                    
                    features = to_var(features, args.device)
                    
                    targets.append(truth_data.numpy())
                    truth_data = to_var(truth_data, args.device)
                    with torch.set_grad_enabled(phase == 'train'):
                        with torch.amp.autocast(args.device):
                            output, loss_1 = model(features, args)                        
                            loss_2 = loss_func(truth=truth_data, predict=output)
                            loss = create_main_loss(loss_1,loss_2,args)
                        
                        if phase == 'train':    
                            optimizer.zero_grad()
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 10.0)
                            scaler.step(optimizer)
                            scaler.update()
                    desc = f"loss1: {loss_1.item()}, loss2: {loss_2.item()}"
                    tqdm_loader.set_description(
                        f'{phase} epoch: {epoch}, {phase} loss: {(running_loss[phase] / steps) :.8f}, '
                        + desc
                    )
                    with torch.no_grad():
                        predictions.append(output.cpu().detach().numpy())

                    running_loss[phase] += loss.item() * truth_data.size(0)
                    # if step % 1000 == 0:
                    #     torch.cuda.empty_cache()
                    #     gc.collect()

                torch.cuda.empty_cache()
                gc.collect()
                    
                predictions = np.concatenate(predictions).copy()
                targets = np.concatenate(targets).copy()
                
                # assert predictions[0].shape == targets[0].shape, f'{predictions.shape}, {targets.shape}'
                
                scores = calculate_metrics(predictions.reshape(predictions.shape[0], -1),
                                           targets.reshape(targets.shape[0], -1), args, plot=epoch % 5 == 0, **kwargs)
                with open(model_folder+"/output.txt", "a") as f:
                    f.write(f'{phase} epoch: {epoch}, {phase} loss: {running_loss[phase] / steps}\n')
                    f.write(str(scores))
                    f.write('\n')
                    f.write(str(time.time()))
                    f.write("\n\n")
                print(scores)
                
                msg.append(f"{phase} epoch: {epoch}, {phase} loss: {running_loss[phase] / steps}\n {scores}\n")
                if phase == 'val':
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Current LR: {current_lr:.6e}")
                    if scores['MAE'] < best_mae:
                        best_mae = scores['MAE']
                        save_dict.update(
                            state_dict=copy.deepcopy(model.state_dict()),
                            epoch=epoch,
                            optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
                            # scheduler_state_dict=copy.deepcopy(scheduler.state_dict())
                        )
                        save_model(f"{model_folder}/best_model.pkl", **save_dict)
                        

                    else:

                        print(f"Current MAE {scores['MAE']} more than best MAE {best_mae}")

            # scheduler.step(running_loss['val'])
    finally:
        time_elapsed = time.perf_counter() - since
        print(f"cost {time_elapsed} seconds")

        save_model(f"{model_folder}/best_model.pkl", **save_dict)
        save_model(f"{model_folder}/final_model.pkl",
                   **{'state_dict': copy.deepcopy(model.state_dict()),
                      'epoch': epoch,
                      'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                    #   'scheduler_state_dict': copy.deepcopy(scheduler.state_dict())
                      })