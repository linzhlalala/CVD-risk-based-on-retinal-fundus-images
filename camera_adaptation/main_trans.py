# this edition is for single target regression
# patt_bn
import argparse
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F

from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from modules.pair_camera_att_cross import PairedDataset
import warnings
from modules.att_block import MyBlock
warnings.filterwarnings(action='once')

parser = argparse.ArgumentParser(description='Retina Health Score Training')
parser.add_argument('--start-epoch', default=0, type=int, 
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, 
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--print-freq', default=10, type=int,
                    help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--base', default='', type=str,
                    help='path to base model (for fine-tune)')
parser.add_argument('--pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=1024, type=int,
                    help='seed for initializing training. 1024 for reproductivity')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')  
#debugging args, change it on other device
parser.add_argument('--workers', default=0, type=int, 
                    help='number of data loading workers (default: 0)')
parser.add_argument('--batch-size', default=32 , type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--epochs', default=200, type=int, 
                    help='number of total epochs to run')
parser.add_argument('--id', default='rgs', type=str, 
                    help='job id to save logs')
parser.add_argument('--backbone', default='condition', type=str, 
                    help='the cached of backbone model')

checkpoint_path = ''

labels = ['WHO-CVD', 'Age','SBP','Cholesterol','BMI', # Regression
    'Gender','Smoker','dmstatus'] #Classification
rgs_divider = 5


# labels = ['WHO-CVD', 'Age', # Regression
#     'Gender'] #Classification
# rgs_divider = 2
    
class tfc(nn.Module):
    def __init__(self):
        super(tfc, self).__init__()
        dim = 384
        
        self.attblock = nn.Sequential(MyBlock())
        self.trans = nn.Sequential(nn.Linear(dim,dim), #,bias=False
                                nn.BatchNorm1d(dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(dim, dim), #,bias=False
                                nn.BatchNorm1d(dim),
                                nn.ReLU(inplace=True), 
                                nn.Linear(dim, dim))
        #self.fc = nn.Sequential(nn.Linear(384, 3))
        self.fc = nn.Linear(384, 8)
        for param in self.fc.parameters():
            param.require_grad = False
            
    def forward(self, input):
        fc1 = self.attblock(input)
        fc2 = self.trans(input[:,0])
        fc2 = fc1[:,0]
        res = self.fc(fc2)
        return fc1, fc2, res
    
def main():
    global checkpoint_path
    args = parser.parse_args()
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)       
        np.random.seed(args.seed)

        cudnn.deterministic = True
        print("Using seed : {args.seed}".format)

    checkpoint_path = 'logs/'+args.id

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    log_file = os.path.join(checkpoint_path,"logs.csv")
    if os.path.exists(log_file):
        os.rename(log_file,log_file.replace(".csv","_backup.csv"))
   
    model = tfc()
    
    predictor = torch.load('data/{}_fc.tar'.format(args.backbone), map_location='cpu')
    model.fc.load_state_dict(predictor)
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define loss function and optimizer
    init_lr = args.lr
    optimizer = torch.optim.Adam([{'params':model.attblock.parameters()},
                                  {'params':model.trans.parameters()}], init_lr,
                                weight_decay=args.weight_decay)
    cudnn.benchmark = True
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            # if args.gpu is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    # Data loading and split it into train, val, test
    train_dataset = PairedDataset('train',4,"data/{}".format(args.backbone))
    val_dataset = PairedDataset('val',4,"data/{}".format(args.backbone))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,num_workers=0, pin_memory=True, shuffle = True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,num_workers=0, pin_memory=True, shuffle = False)
    # train val and test
    # acc1 actually is mse(l2) at here
    best_acc = 0
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        # no adjust as there is plateau scheduler instead
        adjust_learning_rate(optimizer, epoch, args)

        end = time.time()
        # train for one epoch
        train_loss, train_score = train(train_loader, model, optimizer, epoch, args)

        val_loss , val_score  = validate(val_loader, model, epoch, args)
        
        benchmark = val_score['WHO-CVD_corr']
        is_best = benchmark > best_acc
        if is_best:
            best_acc = benchmark
            best_epoch = epoch

        # Update info to a logging csv
        time_cost = time.time() - end

        epoch_dict = {"epoch":epoch, "train_loss":train_loss, "val_loss":val_loss,
                    "epoch_time":time_cost, "lr": optimizer.param_groups[0]['lr'],
                    "train_score":train_score['WHO-CVD_corr'],
                    "val_score":val_score['WHO-CVD_corr']}

        log_to_file(log_file,epoch_dict)
        # checkpoint
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            },is_best,'checkpoint.pth.tar')
        # renew lr
        if epoch %20 == 0:
            print(epoch)
    print(best_acc,'@',best_epoch)


def train(train_loader, model, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4e')

    # switch to train mode
    model.train()
    
    end = time.time()
    
    idxs = []
    p1s = []
    p2s = []
    ptran = []
    
    for i, (idx, p1, p2, _) in enumerate(train_loader):
        # measure data loading time
        if args.gpu is not None:
            p1 = p1.cuda(args.gpu, non_blocking=True)
            p2 = p2.cuda(args.gpu, non_blocking=True)
        # compute output
        fc1,fc2, res = model(p1)
        p2_res = model.fc(p2[:,0])
        p1_res = model.fc(p1[:,0])
        #print(output.shape, target.shape)
        loss = F.mse_loss(fc2, p2[:,0]) + F.mse_loss(res[:,0], p2_res[:,0])
        # measure accuracy and record loss
        losses.update(loss.item(), p1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        idxs.append(idx.numpy().flatten())
        p1s.append(p1_res.detach().cpu().numpy())
        p2s.append(p2_res.detach().cpu().numpy())
        ptran.append(res.detach().cpu().numpy())

    index_arr = np.concatenate(idxs,axis=0).reshape(-1,1)
    preds1_arr = np.concatenate(p1s,axis=0)
    preds2_arr = np.concatenate(p2s,axis=0)
    pred_trans_arr = np.concatenate(ptran,axis=0)

    cols = ['patient_id'] + [x+'_p1' for x in labels] + [x+'_p2' for x in labels] + [x+'_ptrans' for x in labels]
    df = pd.DataFrame(data = np.concatenate((index_arr,preds1_arr,preds2_arr, pred_trans_arr),axis=1),columns = cols)
    
    df.to_csv(os.path.join(checkpoint_path,"train_{}.csv".format(epoch)), index = False)
    
    r2 = {col+'_corr':r2_score(df[col+'_p2'], df[col+'_ptrans']) for col in labels}

    return losses.avg, r2

def validate(val_loader, model,  epoch, args):
    global checkpoint_path
    # switch to evaluate mode
    losses = AverageMeter('Loss', ':.4e')

    model.eval() 
    idxs = []
    p1s = []
    p2s = []
    ptran = []
    with torch.no_grad():
        end = time.time()
        for i, (idx, p1, p2, _) in enumerate(val_loader):
            if args.gpu is not None:
                p1 = p1.cuda(args.gpu, non_blocking=True)
                p2 = p2.cuda(args.gpu, non_blocking=True)

            fc1,fc2, res = model(p1)
            p1_res = model.fc(p1[:,0])
            p2_res = model.fc(p2[:,0])
            loss =  F.mse_loss(fc2, p2[:,0]) + F.mse_loss(res[:,0], p2_res[:,0])
            # measure accuracy and record loss
            losses.update(loss.item(), p1.size(0))
            
            idxs.append(idx.numpy().flatten())
            p1s.append(p1_res.detach().cpu().numpy())
            p2s.append(p2_res.detach().cpu().numpy())
            ptran.append(res.detach().cpu().numpy())
                        
    index_arr = np.concatenate(idxs,axis=0).reshape(-1,1)
    preds1_arr = np.concatenate(p1s,axis=0)
    preds2_arr = np.concatenate(p2s,axis=0)
    pred_trans_arr = np.concatenate(ptran,axis=0)
    

    cols = ['patient_id'] + [x+'_p1' for x in labels] + [x+'_p2' for x in labels] + [x+'_ptrans' for x in labels]
    df = pd.DataFrame(data = np.concatenate((index_arr,preds1_arr,preds2_arr, pred_trans_arr),axis=1),columns = cols)
    
    df.to_csv(os.path.join(checkpoint_path,"val_{}.csv".format(epoch)), index = False)
    
    
    r2 = {col+'_corr':r2_score(df[col+'_p2'], df[col+'_ptrans']) for col in labels}
    
    # p1s = np.concatenate(p2s+ptran, axis= 0)

    return losses.avg, r2


def log_to_file(log_file,epoch_dict):
    """log training infomation to csv for better display"""
    #global checkpoint_path
    #log_file = os.path.join(checkpoint_path,"logs.csv")

    crt_time = time.asctime(time.localtime(time.time()))
    epoch_dict['time'] = crt_time

    if not os.path.exists(log_file):
        record_table = pd.DataFrame()
    else:
        record_table = pd.read_csv(log_file)
    record_table = record_table.append(epoch_dict, ignore_index=True)
    record_table.to_csv(log_file, index=False)

def save_checkpoint(state, is_best, filename):
    #current
    global checkpoint_path
    torch.save(state, os.path.join(checkpoint_path,filename))
    #best
    if is_best:
        shutil.copyfile(os.path.join(checkpoint_path,filename),os.path.join(checkpoint_path,filename.replace('checkpoint','model_best')))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.75 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()