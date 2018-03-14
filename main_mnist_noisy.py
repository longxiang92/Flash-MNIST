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
from torchvision import transforms
from mnist_noisy import MNISTNoisy
from torch.autograd import Variable
import config

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for testing (default: 1024)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and tcuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    tcuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = DataLoader(
    MNISTNoisy(config.data_dir, train=True, generate=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))])
              ),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(
    MNISTNoisy(config.data_dir, train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))])
              ),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


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
        self.fc2 = nn.Linear(50, 11)

    def forward(self, x):
        x = self.extractor(x)
        x = self.fc2(x)
        return F.log_softmax(x)
        

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
        output = model(data)
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

def test():
    model.eval()
    test_loss = 0
    correct = 0
    wrong_idx = []
    for data, target, idxs in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1)[1] # get the index of the max log-probability
        corrects = pred.eq(target.data.view_as(pred)).cpu()
        correct += corrects.sum()
        for idx, c in zip(idxs, corrects.numpy()):
            if not c:
                wrong_idx.append(idx)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
#     print wrong_idx[:10]
    return 100. * correct / len(test_loader.dataset)

def save_checkpoint(epoch):
    state = model.extractor.state_dict()
    file_path = config.get_model_path('extractor_%03d'%epoch)
    print 'save checkpoint: %s'%file_path
    torch.save(state, file_path)
       
for epoch in range(1, args.epochs + 1):
    train(epoch)
    acc = test()
    save_checkpoint(epoch)
    