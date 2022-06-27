from __future__ import print_function, division, absolute_import
import torch
from torch.functional import Tensor
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import timm 



class vit_mt(nn.Module):
    def __init__(self, opt=None):
        super(vit_mt, self).__init__()

        self.img_model = timm.create_model('vit_small_r26_s32_384', num_classes=0,pretrained=True)
        self.fc = nn.Linear(384,8)

    def forward(self, img):
        # batch
        B = img.shape[0]
        # reshape
        x = torch.flatten(img, start_dim=0, end_dim=1)

        z = self.img_model(x)
        # predict
        pred = self.fc(z)        
        return z, pred
    

if __name__ == '__main__':
    model =  vit_mt()
    print('success')

    img = torch.rand(8,4,3,384,384)
    x, avg, res = model(img)
    print(x.shape, avg.shape)