# 量化和微调
import numpy as np
import math
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim

batch_size = 64
test_batch_size = 512
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


# 量化权重
def quantization(weights, bw, fl):
    v_max = (pow(2, bw-1)-1)*pow(2, -fl)
    v_min = -1*pow(2, bw-1)*pow(2, -fl)
    max_out_of_range = (weights>v_max).float()
    min_out_of_range = (weights<v_min).float()
    weights *= 1-max_out_of_range
    weights *= 1-min_out_of_range
    weights += max_out_of_range*v_max
    weights += min_out_of_range*v_min
    weights = torch.round(weights/pow(2, -fl))
    return weights*pow(2, -fl)

# 自定义量化操作

class quantization_F(Function):
    @staticmethod
    def forward(ctx, input, bw, fl):
        v_max = (pow(2, bw-1)-1)*pow(2, -fl)
        v_min = -1*pow(2, bw-1)*pow(2, -fl)
        max_out_of_range = (input>v_max).float()
        min_out_of_range = (input<v_min).float()
        input *= 1-max_out_of_range
        input *= 1-min_out_of_range
        input += max_out_of_range*v_max
        input += min_out_of_range*v_min
        input = torch.round(input/pow(2, -fl))
        return input*pow(2, -fl)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

# 记录数据流的字典
feature_maps = {}

# 目标网络 eval_flag 在需要获取数据流时，需设为1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,  20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)

        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(50*4*4, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x, eval_flag = False):
        in_size = x.size(0)
        if eval_flag:
            feature_maps['input'] = x
            x = F.relu(self.mp(self.conv1(x)))
            feature_maps['conv1'] = x
            x = F.relu(self.mp(self.conv2(x)))
            feature_maps['conv2'] = x
            x = x.view(in_size, -1)
            x = F.relu(self.fc1(x))
            feature_maps['fc1'] = x
            x = self.fc2(x)
            feature_maps['fc2'] = x
        else:
            x = F.relu(self.mp(self.conv1(x)))
            x = F.relu(self.mp(self.conv2(x)))

            x = x.view(in_size, -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        return x
model = Net().cuda()
checkpoint = torch.load('lenet.pth.tar')
for k,v in checkpoint['state_dict'].items():
    print(k)
model.load_state_dict(checkpoint['state_dict'])
acc = test(model)

# 统计权重的最大值
d = model.state_dict()
stored_w = {}
for k,w in d.items():
    w = w.view(1,-1).abs()
    stored_w[k] = [(w[0,:]).max(), (w[0,:]).min()]
for k,v in stored_w.items():
    print(k,'max:',v[0], 'min:',v[1])

# 计算权重的位数
int_w = {}
for k,w in stored_w.items():
    int_w[k] = int(1+ np.ceil(np.log2(w[0].cpu())))
print(int_w)

# 计算量化后的权重
q_state_dict = {}
bit_width = 8
for k,w in d.items():
    q_state_dict[k] = quantization(w, bit_width, bit_width-int_w[k])

# 计数据流中最大的数值
model.eval().cuda()
blob_data = {}
for images, labels in test_loader:
    test = Variable(images.view(-1,1,28,28)).cuda()
    outputs = model(test, eval_flag = True)
    for k in feature_maps.keys():
        tmp_max = feature_maps[k].max().cpu().data.numpy()
        if k in blob_data:
            if tmp_max>blob_data[k]:
                blob_data[k] = tmp_max
        else:
            blob_data[k] = tmp_max
    feature_maps.clear()

# 计算数据流的位数
int_blob = {}
for k,b in blob_data.items():
    # print(b)
    int_blob[k] = np.ceil(np.log2(b)+1)

model.load_state_dict(q_state_dict)
model.eval()
test_loss = 0
correct = 0
for data, target in test_loader:
    data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    test_loss += F.cross_entropy(output, target, size_average=False).item()
    # get the index of the max log-probability
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

test_loss /= len(test_loader.dataset)
print('quantization weights: ')
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

class LeNet_quantization(nn.Module):
    def __init__(self):
        super(LeNet_quantization, self).__init__()
        self.conv1 = nn.Conv2d(1,  20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)

        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(50*4*4, 500)
        self.fc2 = nn.Linear(500, 10)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = quantization_F.apply(x, bit_width, bit_width-int_blob['input'])
        x = F.relu(self.mp(self.conv1(x)))
        x = quantization_F.apply(x, bit_width, bit_width-int_blob['conv1'])
        x = F.relu(self.mp(self.conv2(x)))
        x = quantization_F.apply(x, bit_width, bit_width-int_blob['conv2'])
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = quantization_F.apply(x, bit_width, bit_width-int_blob['fc1'])
        x = self.fc2(x)
        x = quantization_F.apply(x, bit_width, bit_width-int_blob['fc2'])
        return x

# 重建网络加载预训练参数
model2 = LeNet_quantization().cuda()
model2.load_state_dict(q_state_dict)
model2.eval()
test_loss = 0
correct = 0
for data, target in test_loader:
    data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    output = model2(data)
    test_loss += F.cross_entropy(output, target, size_average=False).item()
    # get the index of the max log-probability
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

test_loss /= len(test_loader.dataset)
print('quantization all: ')
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

error = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = optim.SGD(model2.parameters(), lr=learning_rate, momentum=0.9) # SGD优化器，设置权重衰减、动量
model2.cuda()
count = 0
num_epoches = 20
best_acc = 0
for epoch in range(num_epoches):
    model2.train()
    if epoch==10:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    for i, (images, labels) in enumerate(train_loader):
        train = Variable(images.view(-1, 1, 28, 28)).cuda()
        labels = Variable(labels).cuda()
        optimizer.zero_grad()
        outputs = model2(train)
        loss = error(outputs, labels)
        model2.load_state_dict(d)
        loss.backward()
        optimizer.step()
        count += 1
        d = model2.state_dict()
        q_state_dict = {}
        # bit_width = 8
        for k, w in d.items():
            q_state_dict[k] = quantization(w, bit_width, bit_width - int_w[k])
        model2.load_state_dict(q_state_dict)
    # 测试
    correct = 0
    total = 0
    model2.eval()
    for images, labels in test_loader:
        test = Variable(images.view(-1, 1, 28, 28)).cuda()
        outputs = model2(test)
        predicted = torch.max(outputs.data, 1)[1]
        total += len(labels)
        correct += (predicted.cpu() == labels).sum()
    accuracy = 100 * correct / float(total)
    print('epoch: {}  Loss: {}  Accuracy: {} %'.format(epoch, loss.item(), accuracy))
    if accuracy>best_acc:
        best_acc = accuracy
print(best_acc)
