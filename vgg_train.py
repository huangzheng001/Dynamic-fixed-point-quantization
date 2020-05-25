from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import math
# from model.vgg import vgg
import shutil
import random
# random.seed(555)
# np.random.seed(555)
# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# torch.backends.cudnn.deterministic = True

batch_size = 128
test_batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Pad(4),
                       transforms.RandomCrop(32),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=test_batch_size, shuffle=True, **kwargs)

def train(epoch):
    model.train()
    acu_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        acu_loss += loss.item()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return loss.item()


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

class vgg_(nn.Module):

    def __init__(self, dataset='cifar10', init_weights=True, cfg=None):
        super(vgg_, self).__init__()
        if cfg is None:
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        # self.feature = self.make_layers(cfg, True)
        num_classes = 10
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                          nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 =nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                          nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                          nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                          nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
                          nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                          nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                          nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
                          nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                          nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.conv10 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                          nn.BatchNorm2d(512), nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                          nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.conv12 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                          nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.conv13 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                          nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self._initialize_weights()
    # def make_layers(self, cfg, batch_norm=False):
    #     layers = []
    #     in_channels = 3
    #     for v in cfg:
    #         if v == 'M':
    #             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    #         else:
    #             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
    #             if batch_norm:
    #                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
    #             else:
    #                 layers += [conv2d, nn.ReLU(inplace=True)]
    #             in_channels = v
    #     return layers

    def forward(self, x):
        # x = self.feature(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        ##################################################
        x = self.conv3(x)
        x = self.conv4(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        ####################################
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        #########################################
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        ############################################
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

model = vgg_()
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

best_prec1 = 0.
epochs = 160
# state_dict = torch.load('model_vgg.pth.tar')
# model.load_state_dict(state_dict['state_dict'])
# test()
r0 = []
r1 = []
for epoch in range(0, epochs):
    if epoch in [epochs*0.5, epochs*0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    loss = train(epoch)
    prec1 = test()
    r0.append(loss)
    r1.append(prec1)
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    if is_best:
      torch.save({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'best_prec1': best_prec1,
          'optimizer': optimizer.state_dict(),
      }, 'model_vgg128.pth.tar')
print(best_prec1)
# np.save('loss_plot.npy',r0)
# np.save('acc_plot.npy', r1)