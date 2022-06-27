import json

import os
import json
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
import argparse
import cv2
from torchvision.transforms.functional import normalize
from torchvision.utils import save_image
import json
import pandas as pd
infos = 'da_data.json'

# old split with macula checked
class PairedDataset(Dataset):
    "Dataset include dataset info loading and tokenizing"
    def __init__(self, mode, split,data_path = 'data/condition'):
        df = pd.read_json(infos)
        self.mode = mode
        if mode == 'train':
            df = df.loc[df['fold']!=split].copy()
        if mode == 'val':
            df = df.loc[df['fold']==split].copy()
        print("LOADING DATA SHAPE:",df.shape)
        
        self.items = []
        for idx, row in df.iterrows():
            c1,c2 = ['mw','topcon']
            photos = row['photos']
            c1g = [p for p in photos if c1 in p]
            c2g = [p for p in photos if c2 in p]
            for p1 in c1g:
                for p2 in c2g:
                    self.items.append({'idx':idx,
                            'p1':np.load(os.path.join(data_path,row['name'],p1.replace('.png','.npy'))),
                            'p2':np.load(os.path.join(data_path,row['name'],p2.replace('.png','.npy'))),
                            'target':np.zeros(8, dtype=np.float32), # not use, just for align with UKB format
                            })
        print("{}:{} split length:{}".format(mode, split,len(self.items)))  
        
    def save_data(self, file):
        df = pd.DataFrame(self.items)
        df.to_csv(file, index = False)
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):       
        x = self.items[idx]
        id,p1,p2,target = x['idx'],x['p1'], x['p2'], x['target']
        
        return id,p1,p2,target
    
if __name__ == '__main__':

    #train = db_UkbDataset(args,'train')
    val = PairedDataset('train', 4)
    
    _,p1,p2,_ = val.__getitem__(0)
    #save_image(img,"test_img.png")
    print(p1.shape,p2.shape)
    print(len(val))
    
    val = PairedDataset('val', 4)
    #val.save_data('val.csv')
    
    _,p1,p2,_ = val.__getitem__(0)
    #save_image(img,"test_img.png")
    print(p1.shape,p2.shape)
    print(len(val))