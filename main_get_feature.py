'''
Created on Oct 30, 2017

@author: longxiang
'''

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda as tcuda
from torch.utils.data import DataLoader
from torchvision import transforms
from mnist_flash import MNISTFlash
from mnist_feature import MNISTFeature
from torch.autograd import Variable
from mnist_sampler import dump_pkl
import config

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for training (default: 1024)')
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for testing (default: 1024)')
parser.add_argument('--epoch', type=int, default=1, metavar='N',
                    help='the epoch used to extract features (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and tcuda.is_available()

class VideoWrap(object):
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, video):
        res = []
        for img in video:
            timg = self.transform(img)
            res.append(timg)
        res = torch.stack(res, 0)
        return res

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = DataLoader(
    MNISTFlash(config.data_dir, train=True, generate=True,
                   transform=VideoWrap(
                       transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                           ])  
                    )
              ),
    batch_size=args.batch_size, shuffle=False, **kwargs)
test_loader = DataLoader(
    MNISTFlash(config.data_dir, train=False,
                   transform=VideoWrap(
                       transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                           ])  
                    )
              ),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x)
        
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.extractor = Extractor()

    def forward(self, x):
#         print x.size()
        batch_size = x.size(0)
        length = x.size(1)
        flatx = x.view(batch_size*length, x.size(2), x.size(3), x.size(4))
        flatx = self.extractor(flatx)
        x = flatx.view(batch_size, length, -1)
#         print x.size()
        return x
        

model = Net()
checkpoint_path = config.get_model_path("extractor_%03d"%args.epoch)
state_dict = torch.load(checkpoint_path)
model.extractor.load_state_dict(state_dict)

if args.cuda:
    model.cuda()

def extract(loader, path):
    model.eval()
    feats = []
    labels = []
    indexs = []
    for i, (data, target, idxs) in enumerate(loader):
        if i % args.log_interval == 0:
            print '%d/%d'%(i, len(loader))
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        for f, t, idx in zip(output.cpu().data.numpy(), target.cpu().data.numpy(), idxs):
            feats.append(f)
            labels.append(t)
            indexs.append(idx)
#             print idx, f.shape, f.dtype

    dump_pkl((feats, labels, indexs), path)
      
extract(train_loader, os.path.join(config.data_dir, MNISTFeature.training_file))
extract(test_loader, os.path.join(config.data_dir, MNISTFeature.test_file))   