import torch.nn as nn
import math

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)

import _init_paths
from common import *
from net.pretrain_model.resnet import *

def softmax_cross_entropy_criterion(logit, truth, is_average=True):
    loss = F.cross_entropy(logit, truth, reduce=is_average)
    return loss

def metric(logit, truth, is_average=True):

    with torch.no_grad():
        prob = F.softmax(logit, 1)
        value, top = prob.topk(3, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        if is_average==True:
            # top-3 accuracy
            correct = correct.float().sum(0, keepdim=False)
            correct = correct/len(truth)

            top = [correct[0], correct[0]+correct[1], correct[0]+correct[1]+correct[2]]
            precision = correct[0]/1 + correct[1]/2 + correct[2]/3
            return precision, top

        else:
            return correct


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):

    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def load_pretrain(self, pretrain_file):
        #raise NotImplementedError
        #self.resnet.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))

        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())

        # from pprint import pprint as pp
        # pp(keys)
        # pp(list(pretrain_state_dict.keys()))

        for key in keys:
            if any(s in key for s in ['num_batches_tracked', 'classifier.1.weight', 'classifier.1.bias']):
                continue
            if any(s in key for s in []):
                continue
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
        print('')

    def forward(self, x):
        batch_size,C,H,W = x.shape
        mean=[0.485, 0.456, 0.406] #rgb
        std =[0.229, 0.224, 0.225]
        x = torch.cat([
            (x[:,[0]]-mean[0])/std[0],
            (x[:,[1]]-mean[1])/std[1],
            (x[:,[2]]-mean[2])/std[2],
        ],1)

        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def set_mode(self, mode, is_freeze_bn=False ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
            if is_freeze_bn==True: ##freeze
                for m in self.modules():
                    if isinstance(m, BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad   = False

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def run_check_net():

    batch_size = 32
    C,H,W = 3, 32, 32
    num_class = 5

    input = np.random.uniform(0,1, (batch_size,C,H,W)).astype(np.float32)
    truth = np.random.choice (num_class,   batch_size).astype(np.float32)

    #------------
    input = torch.from_numpy(input).float().cuda()
    truth = torch.from_numpy(truth).long().cuda()


    #---
    criterion = softmax_cross_entropy_criterion

    net = MobileNetV2(n_class=5).cuda()
    net.load_pretrain('/dataset/pytorch_models/mobilenet_v2.pth')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             # std=[0.229, 0.224, 0.225])

    print(net)
    ## exit(0)

    logit = net(input)
    loss  = criterion(logit, truth)
    precision, top = metric(logit, truth)

    print('loss    : %0.8f  '%(loss.item()))
    print('correct :(%0.8f ) %0.8f  %0.8f '%(precision.item(), top[0].item(),top[-1].item()))
    print('')



    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=0.1, momentum=0.9, weight_decay=0.0001)

    #optimizer = optim.Adam(net.parameters(), lr=0.001)


    i=0
    optimizer.zero_grad()
    print('        loss  | prec      top      ')
    print('[iter ]       |           1  ... k ')
    print('-------------------------------------')
    while i<=500:

        logit   = net(input)
        loss    = criterion(logit, truth)
        precision, top = metric(logit, truth)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()

        if i%20==0:
            print('[%05d] %0.3f | ( %0.3f ) %0.3f  %0.3f'%(
                i, loss.item(),precision.item(), top[0].item(),top[-1].item(),
            ))
        i = i+1

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

    # import torchvision.models as models
    # resnet18 = models.resnet34(pretrained=True)

    print( 'sucessful!')
