import pdb
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.quantization
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import os
import argparse
from models import *
from utils import progress_bar
from video import InputVideo
import cv2
from PIL import Image
from latency import Latency
import numpy as np
import csv 
from thop import profile
from approximate import Approximate


best_prec1 = 0

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int, metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set', default = False)
parser.add_argument('-t', '--train', dest='train', action='store_true', help='evaluate model on validation set', default=False)
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--cpu', dest='cpu', action='store_true', help='use cpu', default = False)
parser.add_argument('--save-dir', dest='save_dir', help='The directory used to save the trained models', default='save_temp', type=str)
parser.add_argument('--model', dest='model_type', help='The model used', default='VGG16', type=str)
parser.add_argument('--benchmarks', dest='benchmarks', action='store_true', help='get benchmarks for model', default=False)
parser.add_argument('--prune', dest='prune', default=0, type=float, help='pruning level')
parser.add_argument('--quant', dest='quant', default = "float32", help='level of quantization')
parser.add_argument('--fps', dest='fps', default=60, type=float, help='input fps%')
parser.add_argument('--sampling_rate', dest='samp_rate', default=1, type=float, help='sensor sampling rate')
parser.add_argument('--test_batch_size', dest='test_bs', default=1, type=int, help='test batchsize')
parser.add_argument('--resolution', dest='res', default=1, type=float, help='input resolution%')                  
parser.add_argument('--output_file', dest='output_csv', help='csv file to save data', default='csv_files/default.csv', type=str)
parser.add_argument('--data_collection', dest='data_collection', action='store_true', help='collecting data?', default=False)

args = parser.parse_args()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
 

def benchmark(net):
    
    net.eval()
    if args.prune:
        net.prune()

    if args.quant != 'float32':
        if args.quant == 'float16':
            net.half()
            criterion.half()
        else:
            net.qconfig = torch.quantization.default_qconfig
            torch.quantization.prepare(net, inplace=True)
            torch.quantization.convert(net, inplace=True)


    input = torch.randn(1, 3, int(224*args.res), int(224*args.res)).to(device)
    macs, params = profile(net, inputs=(input, ))

    if args.data_collection:
        #acc = test(0, 0, net, save = False)
        acc = 1
        length = 4
        video = InputVideo(int(224*args.res),int(224*args.res), args.fps, length, "generated_video.avi")
        video.create_video()
        cap = cv2.VideoCapture("generated_video.avi")
        latency = Latency(cap, net, args.fps, args.samp_rate, args.quant, args.res, length, args.model_type, device)
        count, time_taken = latency.measure_latency()
        cap.release()
        output_fps = (count)/time_taken

    else:
        approximator = Approximate(macs, args.res*224, args.prune, args.quant, args.model_type)
        acc, speedup = approximator.get_approximates()
        length = 4
        video = InputVideo(int(224*args.res),int(224*args.res), args.fps, length, "generated_video.avi")
        video.create_video()
        cap = cv2.VideoCapture("generated_video.avi")
        latency = Latency(cap, net, args.fps, args.samp_rate, args.quant, args.res, length, args.model_type, device)
        count, time_taken = latency.measure_latency()
        cap.release()
        output_fps = (count)/time_taken
        output_fps *= speedup
        acc = max(acc, 1/1000)
        macs = macs*(output_fps)

    print(args.model_type, round(args.prune,2), args.quant, round(args.test_bs,2), round(args.samp_rate,2), round(args.res*224,2), round(acc,3), output_fps, macs)
    to_write = [args.model_type, round(args.prune,2), args.quant, round(args.test_bs,2), round(args.samp_rate*args.fps,2), int(args.res*224), round(acc,3), round(output_fps,2), macs]
    
    with open(args.output_csv, 'a') as csvfile:   
        csvwriter = csv.writer(csvfile)   
        csvwriter.writerow([str(i) for i in to_write])

def test(epoch, best_prec1, net, save = True):
    if args.prune:
        net.prune()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            net.eval()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    prec1 = correct/total
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    if save:
        save_checkpoint({
            'epoch': 1 + 1,
            'state_dict': net.state_dict(),
            'best_prec1': 1,
        }, True, filename = "saved_models/model_{model_name}_{pruning}_{quant}.pth.tar".format(model_name=args.model_type, pruning=args.prune, quant = args.quant))

    if is_best:
        return prec1
    else:
        return best_prec1


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.cpu == True:
    device = 'cpu'

start_epoch = 0

print('==> Preparing data..')

if args.data_collection:
    train_root = '/home/data/ilsvrc/ILSVRC/ILSVRC2012_Classification/train'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_data = datasets.ImageFolder(train_root, transform=transforms.Compose([
            transforms.RandomResizedCrop(int(224*args.res)),
            transforms.ToTensor(),
            normalize
    ]))

    shuffle_dataset = True
    val_split = 0.0009

    dataset_size = len(train_data)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(2)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    val_sampler = SubsetRandomSampler(val_indices)

    testloader = torch.utils.data.DataLoader(train_data,
                sampler = val_sampler,
                batch_size = 25
            )

# Model
print('==> Building model ', args.model_type)

if args.model_type == "VGG16":
    net = vgg16(pruning = args.prune)
if args.model_type == "VGG19":
    net = VGG('VGG19')
if args.model_type == "RESNET":
    net = resnet50(pruning = args.prune)
if args.model_type == "MOBILENET":
    net = mobilenet_v2(pruning = args.prune)
if args.model_type == "INCEPTION":
    net = inception_v3(pruning = args.prune)
if args.model_type == "DENSENET":
    net = densenet121(pruning = args.prune)
if args.model_type == "SHUFFLENET":
    net = shufflenet_v2_x1_0(pruning = args.prune)
if args.model_type == "GOOGLENET":
    net = googlenet(pruning = args.prune)


net = net.to(device)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        net.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.evaluate, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.benchmarks:
    benchmark(net)
    sys.exit()


if args.evaluate:
    if args.quant != 'float32':
        if args.quant == 'float16':
            net.half()
            criterion.half()
        else:
            net.eval()
    best_prec1 = test(0, best_prec1, net)
    sys.exit()

if args.train:
    for epoch in range(start_epoch, start_epoch+args.epochs):
        train(epoch)
        best_prec1 = test(epoch, best_prec1, net)
