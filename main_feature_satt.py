'''
Created on Oct 30, 2017

@author: longxiang
'''

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import cuda as tcuda
from torch.utils.data import DataLoader
from mnist_feature import MNISTFeature
from torch.autograd import Variable
import config
import cPickle

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--natt', type=int, default=10, metavar='N',
                    help='natt')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--logf', type=str, default='log_satt_%d.txt',
                    help='log file path')
args = parser.parse_args()
args.cuda = not args.no_cuda and tcuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    tcuda.manual_seed(args.seed)
    

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = DataLoader(
    MNISTFeature(config.data_dir, train=True,
                   transform=lambda x:torch.from_numpy(x)
              ),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(
    MNISTFeature(config.data_dir, train=False,
                   transform=lambda x:torch.from_numpy(x)
              ),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

import numpy as np
from torch.nn.parameter import Parameter

def get_fans(shape):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out

def normal(shape, scale=0.05):
    tensor = torch.FloatTensor(*shape)
    tensor.normal_(mean = 0.0,  std = scale)
    return tensor

def glorot_normal(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / (fan_in + fan_out))
    return normal(shape, s)

_softmax = nn.Softmax()

def softmax_m1(x):
    flat_x = x.view(-1, x.size(-1))
    flat_y = _softmax(flat_x)
    y = flat_y.view(*x.size())
    return y

class ShiftingAttention(nn.Module):
    def __init__(self, dim, n):
        super(ShiftingAttention, self).__init__()
        self.dim = dim
        self.n_att = n
        
        self.attentions = nn.Conv1d(dim, n, 1, bias=True)
        self.gnorm = np.sqrt(n)
        
        self.w = Parameter(glorot_normal((n,)))
        self.b = Parameter(glorot_normal((n,)))

    def forward(self, x):
        '''x = (N, L, F)'''
        scores = self.attentions(torch.transpose(x, 1, 2))
        '''scores = (N, C, L)'''
        weights = softmax_m1(scores)
        '''weights = (N, C, L), sum(weights, -1) = 1'''
        
        outs = []
        for i in range(self.n_att):
            weight = weights[:,i,:]
            ''' weight = (N, L) '''
            weight = weight.unsqueeze(-1).expand_as(x)
            ''' weight = (N, L, F) '''
            
            w = self.w[i].unsqueeze(0).expand(x.size(0), x.size(-1))
            b = self.b[i].unsqueeze(0).expand(x.size(0), x.size(-1))
            ''' center = (N, L, F) '''
            
            o = torch.sum(x * weight, 1).squeeze(1) * w + b
                            
            norm2 = torch.norm(o, 2, -1, keepdim=True).expand_as(o)
            o = o/norm2/self.gnorm
            outs.append(o)
        outputs = torch.cat(outs, -1)
        '''outputs = (N, F*C)'''
        return outputs, weights


fdim = 50
nclass = 1024
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.att = ShiftingAttention(fdim, args.natt)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(fdim*args.natt, nclass)

    def forward(self, x):
        x, weights = self.att(x)
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x), weights
        
model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()
    correct = 0
    tot = 0
    for batch_idx, (data, target, _) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        tot += len(target)
    
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.2f}%)'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0],
                correct, tot, 100.0*correct/tot))
            correct = 0
            tot = 0

def show_weights(weights):
    s = ''
    for line in weights*100:
        for w in line:
            s += '%6.1f'%w
        s += '\n'
    print s[:-1]
    
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    n_show = 1000
    cnt_show = 0
    memory_show = []
    wrong_idx = []
    for data, target, indexs in test_loader:       
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output, weights = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1)[1] # get the index of the max log-probability
        corrects = pred.eq(target.data.view_as(pred)).cpu()
        correct += corrects.sum()
        for idx, c in zip(indexs, corrects.numpy()):
            if not c:
                wrong_idx.append(idx)
        
        for ws, p, tar, idx in zip(weights.data.cpu().numpy(), pred.cpu().numpy(), target.data.cpu().numpy(), indexs):
            if cnt_show >= n_show:
                break
            if p == tar:
                memory_show.append((idx, ws))
                cnt_show += 1
            
    test_loss /= len(test_loader.dataset)
    print('\nTest Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
#     print wrong_idx[:10]
    print 
    
    with open(args.logf%args.natt, 'a') as f:
        f.write('%.2f\n'%(100. * correct / len(test_loader.dataset)))
        
    return memory_show

def dump_pkl(obj, path):
    f = open(path, 'wb')
    try:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    finally:
        f.close()

def save_weights(weights, epoch):
    path = config.get_output_path('satt%d_%03d.pkl'%(args.natt, epoch))
    dump_pkl(weights, path)

# test()
with open(args.logf%args.natt, 'w'):
    pass
for epoch in range(1, args.epochs + 1):
    train(epoch)
    weights = test(epoch)
    save_weights(weights, epoch)