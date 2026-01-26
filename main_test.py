import sys
import os
import argparse
from test.testing_main import test_main
import torch
import numpy as np
import random
import os
from utils.prepare import load_test_datadict

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metric import calculate_metrics
from utils.util import save_model, to_var
from utils.prepare import create_main_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='input the model name', default='MVVIT_TTE')
    parser.add_argument('-d', '--dataset', type=str, default='porto', help='input the dataset name', choices=['chengdu','porto'])
    parser.add_argument('-i', '--identify', type=str, help='input the specific identification information', default='')

    parser.add_argument('-D', '--device', type=str, help='input the chosen device', default="cuda:0")
    parser.add_argument('-r', '--mask_rate', type=float, help='intput the mask rate of segments in a trajectory', default=0.4)
    parser.add_argument('-s', '--seed', type=int, help='input the seed', default=42)
    args = parser.parse_args()
    args.absPath = os.path.dirname(os.path.abspath(__file__))
    print(args.model)
    print(args.dataset)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    test_main(args)
    sys.exit(0)