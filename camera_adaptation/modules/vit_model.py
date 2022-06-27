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

        self.img_org = timm.create_model('vit_small_r26_s32_384', num_classes=0,pretrained=False)
        self.fc = nn.Sequential(nn.Linear(384,3))

    def forward(self, img):

        x = self.img_org(img)
        res = self.fc(x)
        
        return x,res


if __name__ == '__main__':
    model =  vit_mt()
    print('success')

    img = torch.rand(8,3,384,384)
    profile = torch.rand(8,3,384,384)
    x,res = model(img,profile)
    print(x.shape)
    print(res.shape)