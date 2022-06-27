import torch
import torch.nn as nn
import torch.nn.functional as F

#single
#weights = [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
#multi
#weights = [1.0,1.0,1.0,1.0,1.0,1.1,1.0,1.0]
#clfa
weights = [4.0,1.0,0.0,0.0,0.0,1.0,1.0,1.0]
weights = [x/sum(weights) for x in weights]

class Mix_Loss(nn.Module):
    def __init__(self):
        super(Mix_Loss, self).__init__()
        self.register_buffer("weights", torch.Tensor(weights))
        self.weights.requires_grad = False

    def mt_supervised(self, pred, target, divide = 5):
        
        # The regression and classification loss
        loss_temp = []
        for colume in range(divide):
            c_p = pred[:,colume]
            c_y = target[:,colume]
            loss_temp.append(F.mse_loss(c_p,c_y,reduction='none'))

        for colume in range(divide, pred.size()[1]):
            c_p = pred[:,colume]
            c_y = target[:,colume]
            loss_temp.append(F.binary_cross_entropy_with_logits(c_p,c_y,reduction='none'))
        loss_temp = torch.stack(loss_temp)
        # loss, the rgs+cls loss
        loss = torch.transpose(loss_temp,0,1).matmul(self.weights)
        return loss

    def forward(self, z, pred, target, divide = 5):
        # supervised loss
        pred = pred.view(-1,2,8)
        loss_s1 = self.mt_supervised_penalty(pred[:,0,:],target)
        loss_s2 = self.mt_supervised_penalty(pred[:,1,:],target)
        
        loss1 = torch.mean(loss_s1 + loss_s2)
        # the feature alignment loss  
        better = loss_s1 - loss_s2
        y = z.clone()
        y.retain_grad()
        for i in range(better.shape[0]):
            if better[i] < 0: # 1 is better, need swap
                y[[2*i, 2*i+1]] = z[[2*i+1,2*i]]
        z = y.view(-1,2,384)        
        z1 = z[:,0,:]
        z2 = z[:,1,:]
        loss2 = F.mse_loss(z1,z2.detach())
        
        return loss1 , loss2

if __name__ == '__main__':
    z = torch.rand(36,384)
    p = torch.rand(36,384)
    x = torch.rand(36,8)
    y = torch.ones(18,8)
    criterion = Mix_Loss()
    #print(x1, y)
    print(criterion(z,p,x,y))
