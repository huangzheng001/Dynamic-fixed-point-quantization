# -------------------------------------------------------------
# made by huangzheng
# 2020/06/03
# ------------------------------------------------------------
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
import sys
import warnings
warnings.filterwarnings('ignore')

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


class vgg_merge(nn.Module):

    def __init__(self, dataset='cifar10', init_weights=True, cfg=None):
        super(vgg_merge, self).__init__()
        # if cfg is None:
        #     cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        # self.feature = self.make_layers(cfg, True)
        num_classes = 10
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True),
                          nn.ReLU(inplace=True))
        self.conv2 =nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                          nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
                          nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
                          nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True),
                          nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
                          nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
                          nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True),
                          nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
                          nn.ReLU(inplace=True))
        self.conv10 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
                          nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
                          nn.ReLU(inplace=True))
        self.conv12 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
                          nn.ReLU(inplace=True))
        self.conv13 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
                          nn.ReLU(inplace=True))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, eval_flag = False):
        # x = self.feature(x)
        if eval_flag:
            feature_maps['input'] = x
        x = self.conv1(x)
        if eval_flag:
            feature_maps['conv1'] = x
        x = self.conv2(x)
        if eval_flag:
            feature_maps['conv2'] = x
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        ##################################################
        x = self.conv3(x)
        if eval_flag:
            feature_maps['conv3'] = x
        x = self.conv4(x)
        if eval_flag:
            feature_maps['conv4'] = x
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        ####################################
        x = self.conv5(x)
        if eval_flag:
            feature_maps['conv5'] = x
        x = self.conv6(x)
        if eval_flag:
            feature_maps['conv6'] = x
        x = self.conv7(x)
        if eval_flag:
            feature_maps['conv7'] = x
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        #########################################
        x = self.conv8(x)
        if eval_flag:
            feature_maps['conv8'] = x
        x = self.conv9(x)
        if eval_flag:
            feature_maps['conv9'] = x
        x = self.conv10(x)
        if eval_flag:
            feature_maps['conv10'] = x
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        ############################################
        x = self.conv11(x)
        if eval_flag:
            feature_maps['conv11'] = x
        x = self.conv12(x)
        if eval_flag:
            feature_maps['conv12'] = x
        x = self.conv13(x)
        if eval_flag:
            feature_maps['conv13'] = x
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

model = vgg_merge()
old_model = vgg_()
old_model.cuda()
checkpoint = torch.load('model_vgg128.pth.tar')
state = checkpoint['state_dict']
old_model.load_state_dict(state)
# ============================================
# extra and merge BN
print('extra and merge BN')
new_state = {}
for name, m in old_model.named_modules():
    if isinstance(m, nn.Conv2d):
        w_conv = m.weight.data
        b_onv = 0
        conv_name = name
    if isinstance(m, nn.BatchNorm2d):
        bn_mean = m.running_mean
        beta = m.weight.data.clone()
        gamma = m.bias.data
        var_sqrt = torch.sqrt(m.running_var + m.eps)
        w_shape = w_conv.shape
        tmp = (beta / var_sqrt).view((w_shape[0], 1, 1, 1))
        tmp_w = w_conv * tmp
        tmp_b = (b_onv - bn_mean) / var_sqrt * beta + gamma
        new_state[conv_name+'.weight'] = tmp_w
        new_state[conv_name+'.bias'] = tmp_b
    if isinstance(m, nn.Linear):
        new_state[name+'.weight'] = m.weight.data
        new_state[name+'.bias'] = m.bias.data

model.load_state_dict(new_state)
model.cuda()
print('before:')
test(old_model)
print('after merge BN:')
test(model)

# ==============================
# 统计权重的最大值
d = model.state_dict()
stored_w = {}
for k, w in d.items():
    w = w.view(1, -1).abs()
    stored_w[k] = [(w[0, :]).max(), (w[0, :]).min()]
# for k, v in stored_w.items():
#     print(k, 'max:', v[0], 'min:', v[1])

# 计算权重的位数
int_w = {}
for k, w in stored_w.items():
    int_w[k] = int( np.ceil(np.log2(w[0].cpu())+1))
# print(int_w)

# 计算量化后的权重
q_state_dict = {}
bit_width = 8
activity_width = 8
# bit_width = int(sys.argv[1])
for k, w in d.items():
    q_state_dict[k] = quantization(w.float(), bit_width, bit_width - int_w[k])


# 计数据流中最大的数值
model.eval().cuda()
blob_data = {}
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)
    outputs = model(data, eval_flag=True)
    for k in feature_maps.keys():
        tmp_max = feature_maps[k].abs().max().cpu().data.numpy()
        if k in blob_data:
            if tmp_max > blob_data[k]:
                blob_data[k] = tmp_max
        else:
            blob_data[k] = tmp_max
    feature_maps.clear()
    # break

# 计算数据流的位数
int_blob = {}
for k, b in blob_data.items():
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
print('quantization weights:')
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

class vgg_quantization(nn.Module):

    def __init__(self, dataset='cifar10', init_weights=True, cfg=None):
        super(vgg_quantization, self).__init__()

        num_classes = 10
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=False))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=False))

        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=False))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=False))

        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=False))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=False))
        self.conv7 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=False))

        self.conv8 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=False))
        self.conv9 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=False))
        self.conv10 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
                                    nn.ReLU(inplace=False))

        self.conv11 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
                                    nn.ReLU(inplace=False))
        self.conv12 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
                                    nn.ReLU(inplace=False))
        self.conv13 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
                                    nn.ReLU(inplace=False))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, if_q=False):
        # x = self.feature(x)
        # if if_q:
        x = quantization_F.apply(x, activity_width, activity_width - int_blob['input'])
        x = self.conv1(x)
        x = quantization_F.apply(x, activity_width, activity_width - int_blob['conv1'])
        x = self.conv2(x)
        x = quantization_F.apply(x, activity_width, activity_width - int_blob['conv2'])
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        #################################################
        x = self.conv3(x)
        x = quantization_F.apply(x, activity_width, activity_width - int_blob['conv3'])
        x = self.conv4(x)
        x = quantization_F.apply(x, activity_width, activity_width - int_blob['conv4'])
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        ####################################
        x = self.conv5(x)
        x = quantization_F.apply(x, activity_width, activity_width - int_blob['conv5'])
        x = self.conv6(x)
        x = quantization_F.apply(x, activity_width, activity_width - int_blob['conv6'])
        x = self.conv7(x)
        x = quantization_F.apply(x, activity_width, activity_width - int_blob['conv7'])
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        #########################################
        x = self.conv8(x)
        x = quantization_F.apply(x, activity_width, activity_width - int_blob['conv8'])
        x = self.conv9(x)
        x = quantization_F.apply(x, activity_width, activity_width - int_blob['conv9'])
        x = self.conv10(x)
        x = quantization_F.apply(x, activity_width, activity_width - int_blob['conv10'])
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        ############################################
        x = self.conv11(x)
        x = quantization_F.apply(x, activity_width, activity_width - int_blob['conv11'])
        x = self.conv12(x)
        x = quantization_F.apply(x, activity_width, activity_width - int_blob['conv12'])
        x = self.conv13(x)
        x = quantization_F.apply(x, activity_width, activity_width - int_blob['conv13'])
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        # else:
        #     x = self.conv1(x)
        #     x = self.conv2(x)
        #     x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        #     #################################################
        #     x = self.conv3(x)
        #     x = self.conv4(x)
        #     x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        #     ####################################
        #     x = self.conv5(x)
        #     x = self.conv6(x)
        #     x = self.conv7(x)
        #     x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        #     #########################################
        #     x = self.conv8(x)
        #     x = self.conv9(x)
        #     x = self.conv10(x)
        #     x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        #     ############################################
        #     x = self.conv11(x)
        #     x = self.conv12(x)
        #     x = self.conv13(x)
        #     x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = x.view(x.size(0), -1)

        y = self.classifier(x)

        return y


# 重建网络加载预训练参数
model2 = vgg_quantization().cuda()
model2.load_state_dict(q_state_dict)
model2.eval()
test_loss = 0
correct = 0
for data, target in test_loader:
    data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    output = model2(data, if_q=True)
    test_loss += F.cross_entropy(output, target, size_average=False).item()
    # get the index of the max log-probability
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

test_loss /= len(test_loader.dataset)
print('quantization all: ')
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

optimizer = optim.SGD(model2.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
best_prec1 = 0.
epochs = 40 #200
print('bit width: ', bit_width)
for epoch in range(0, epochs):
#     if epoch in [epochs*0.25, epochs*0.75]:
#     # if epoch in [epochs * 0.5, ]:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] *= 0.1
    # train
    model2.train()
    acu_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model2(data, if_q=True)
        loss = F.cross_entropy(output, target)
        model2.load_state_dict(d)
        loss.backward()
        optimizer.step()
        acu_loss += loss.item()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        d = model2.state_dict()
        q_state_dict = {}
        # bit_width = 8
        for k, w in d.items():
            q_state_dict[k] = quantization(w.float(), bit_width, bit_width - int_w[k])
            # if len(w.shape) == 4 or 'classifier' in k:
            #     q_state_dict[k] = quantization(w.float(), bit_width, bit_width - int_w[k])
            # else:
            #     q_state_dict[k] = w
        model2.load_state_dict(q_state_dict)
    # # test
    model2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model2(data, if_q=True)
        test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    best_prec1 = max(best_prec1, 100. * correct / len(test_loader.dataset))

print('bit width: {},  activity width: {} best: {}'.format(bit_width, activity_width, best_prec1))
