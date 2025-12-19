import sys
import os
import argparse
from train.training_main import test_model
from utils.prepare import create_model
from models.OSM_BE_Resnet_TTE import OSM_BER_TTE
from utils.prepare import load_datadoct_pre, load_test_datadict
import torch
import numpy as np
import random
import os
import json
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='input the model name', default='OSM_BER_TTE_CA_precomputed')
    parser.add_argument('-d', '--dataset', type=str, default='porto', help='input the dataset name', choices=['chengdu','porto'])
    parser.add_argument('-i', '--identify', type=str, help='input the specific identification information', default='')

    parser.add_argument('-D', '--device', type=str, help='input the chosen device', default="cuda:0")
    parser.add_argument('-o', '--optim', type=str, help='input the chosen optimization function', default="Adam", choices=['Adam','AdamW'])
    parser.add_argument('-E', '--epoch_cycle', type=int, help='input the epoch cycle for discriminator training', default=1)
    parser.add_argument('-c', '--loss', type=str, help='input the chosen loss function', default="smoothL1", choices=['rmse','mse', 'mape', 'mae', 'smoothL1'])
    parser.add_argument('-cl', '--loss_val', type=float, help='intput the specific parameter for smoothL1',  default=300.)
    parser.add_argument('-e', '--epochs', type=int, help='input the max epochs',default=50)
    parser.add_argument('-b', '--beta', type=float, help='intput the learning preference between MSG and TTE (the bigger the value, the more preference for TTE.)',default=0.7)
    parser.add_argument('-l', '--lr', type=float, help='intput the initial learning rate',default=0.001)
    parser.add_argument('-w', '--weight_decay', type=float, help='intput the weight decay of optimization',default=0.00001)
    parser.add_argument('-p', '--patience', type=int, help='intput the max iteration times of early stop',default=10)
    parser.add_argument('-r', '--mask_rate', type=float, help='intput the mask rate of segments in a trajectory', default=0.4)
    parser.add_argument('-s', '--seed', type=int, help='input the seed', default=42)
    args = parser.parse_args()
    args.absPath = os.path.dirname(os.path.abspath(__file__))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    absPath = os.path.dirname(os.path.abspath(__file__))

    load_datadoct_pre(args)
    test_loader, scaler = load_test_datadict(args)
    
    model = create_model(args)
    
    model_folder = f'{args.absPath}/data/save_models/{args.model}_{args.identify}_{args.dataset}'
    args.model_folder = model_folder
    
    model.eval()
    model.to(args.device)

    best_model = torch.load(os.path.join(model_folder, 'best_model.pkl'), map_location=args.device)
    model.load_state_dict(best_model['state_dict'], strict=False)
    test_model(model, test_loader, args)