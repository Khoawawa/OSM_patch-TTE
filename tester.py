import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
class Datadict(Dataset):
    def __init__(self, inputs):
        self.content = inputs

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(self.content)
def test_collate(data):
    print(len(data))
    time = torch.Tensor([d[-1] for d in data])
    linkids = []
    dateinfo = []
    inds = []
    for _, l in enumerate(data):
        linkids.append(np.asarray(l[1]))
        dateinfo.append(l[2:5])
        inds.append(l[0])
    lens = np.asarray([len(k) for k in linkids], dtype=np.int16)
    print(linkids[0].shape)
    print(len(linkids))
    def fake_info(xs, date):
        dim = 7 + len(date)
        L = len(xs)
        return [np.zeros(dim, dtype=np.float32) for _ in range(L)]
    
    con_links = np.concatenate([fake_info(b, dateinfo[ind]) for ind, b in enumerate(linkids)], dtype='object')
    
    lat, lon = con_links[:, 6], con_links[:, 7] # start lat lon
    print(lat.shape, lon.shape)
    return {}

tdata = np.load('mydata/test.npy', allow_pickle=True)
loader = DataLoader(Datadict(tdata), batch_size=32,
                                        collate_fn=lambda x: test_collate(x),
                                        shuffle=False, pin_memory=True)
next(iter(loader))