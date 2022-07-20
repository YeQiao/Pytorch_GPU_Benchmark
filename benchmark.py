import os
import argparse
import numpy as np
import time, datetime
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchsummary import summary
from utils import *
from resnet import *
import torchvision.models as models
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

def train_fp16(args, model, epoch, train_loader, scaler, device, optimizer):
    model.train()
    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with autocast():
            output = model(data)
            loss = F.cross_entropy(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def train_fp32(args, model, epoch, train_loader, device, optimizer):
    model.train()
    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, test_loader, device, scheduler = None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    if scheduler is not None:
        scheduler.step(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

def fit(args, model, train_loader,test_loader, device):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    # scheduler = None
    if args.mix_precision:
        scaler = GradScaler()
    best_prec1 = 0.
    for epoch in range(args.epochs):
        if args.mix_precision:      
            train_fp16(args, model, epoch, train_loader, scaler, device, optimizer)
        else:
            train_fp32(args, model, epoch, train_loader, device, optimizer)
        prec1 = test(args, model, test_loader, device, scheduler)
        best_prec1 = max(prec1, best_prec1)

def main():
    model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch Benchmark Training')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names, help='model architecture: ' +
                        ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
    parser.add_argument('--mix_precision', dest='mix_precision', action='store_true',
                    help='use mix_precision model')
    parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
#         model = models.__dict__[args.arch]()
#         if args.dataset == 'cifar10':
#             model.fc = nn.Linear(512, 10)
#         else:
#             model.fc = nn.Linear(512, 100)
        model = ResNet18()
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
    device = 'cuda'
    model.to(device)
    if args.dataset == 'cifar10':
        data_loader = Cifar10DataLoader(batch_size = args.batch_size)
    else:
        data_loader = Cifar100DataLoader(batch_size = args.batch_size)

    train_loader = data_loader['train']
    test_loader = data_loader['val']

    start_t = time.time()
    fit(args, model, train_loader,test_loader, device)
    t_m, t_s = divmod(time.time() - start_t, 60)
    t_h, t_m = divmod(t_m, 60)
    print("Finished training.")
    print('total training time is {:d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s)))

if __name__ == "__main__":
    main()
    # python3 benchmark.py --gpu 1 --mix_precision
