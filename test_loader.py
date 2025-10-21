import utils.prepare as prep
from utils.prepare import load_datadict, build_grid_index
from sklearn.preprocessing import StandardScaler
import os
import json
import pickle
from types import SimpleNamespace
from models.ContextEncoder import ContextEncoder
from models.VideoMae import ViTEncoder
from models.MVVIT_TTE import CrossAttention
from utils.util import to_var
grid_index, edgeinfo, nodeinfo, scaler, scaler2 = None, None, None, None, None
args = SimpleNamespace(
    dataset='porto',
    datapath='./mydata',
    absPath='.',
    mode='train',
    mask_rate=0.4,
    device='cpu'
    
)
abspath = 'utils/data_config.json'

with open(abspath) as file:
    data_config = json.load(file)[args.dataset] 
    args.data_config = data_config
with open(os.path.join(args.absPath,args.data_config['edges_dir']), 'rb') as f:
    edgeinfo = pickle.load(f)
with open(os.path.join(args.absPath,args.data_config['nodes_dir']), 'rb') as f:
    nodeinfo = pickle.load(f)
with open(os.path.join(args.absPath,args.data_config['patch']['patch_dir'],'patch_metadata.json'), 'r') as f:
    patch_json = json.load(f)

scaler = StandardScaler()
scaler.fit([[0, 0]])
scaler.mean_ = [107.497195, 3010.37456]
scaler.scale_ = [131.102877, 2750.78118]
scaler2 = StandardScaler()
scaler2.fit([[0, 0, 0, 0]])
scaler2.mean_ = [-8.62247695, 41.15923239, -8.62256569, 41.15929004]
scaler2.scale_ = [0.02520552, 0.01236445, 0.02526226, 0.01242564]
transform = prep.get_transform()
grid_index = build_grid_index(patch_json, args.data_config['patch']['patch_size'])
prep.info_all = [transform,grid_index,edgeinfo, nodeinfo, scaler, scaler2]

loader,scaler_temp = load_datadict(args)

data, _ = next(iter(loader['train']))
features = to_var(data, 'cpu')
print(type(features['patches'][0]))
print(type(features['mask']))
# visual_model = ViTEncoder()
# context_model = ContextEncoder(8,64,26529+1,4)
# ca_model = CrossAttention(dim_q=visual_model.hidden_size, dim_kv=context_model.hidden_size, num_heads=8)

# visual_output = visual_model(data['patches'])
# ctx_output,_,_ = context_model(data, args)
# cross_attn_output = ca_model(visual_output, ctx_output)
# print("Visual output shape: ", visual_output.shape)  # Expected: [B, T, D]
# print("Context output shape: ", ctx_output.shape)  # Expected: [B, T, D']
# print("Cross Attention output shape: ", cross_attn_output.shape)  # Expected: [B, T, D + D']
# model = ContextEncoder(8, 512, 0, 4)
# output = model(data, args)
# print("Output shape: ", output.shape)  # Expected output shape based on model design

