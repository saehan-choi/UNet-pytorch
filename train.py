import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.data import SummaryWriter
# 여기선 안되네용 ㅎㅎ..

from torchvision import transforms, datasets
import matplotlib.pyplot as plt
# 트레이닝 파라메터 설정
lr = 1e-3
batch_size = 4
num_epoch = 100

data_dir = './datasets'
ckpt_dir = './checkpoint'
log_dir = './log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            # Convolution Batchnormalization Relu 2d
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]

            cbr = nn.Sequential(*layers)

            return cbr

        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        # encoding convolution
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)
        # decoding convolution

        self.uppool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                            kernel_size=2, stride=2, padding=0, bias=True)
            
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.uppool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                            kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.uppool2 = nn.ConvTranspose2d(in_channels= 64, out_channels=64,
                                            kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.uppool1 = nn.ConvTranspose2d(in_channels=32, out_channels=32,
                                            kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.uppool4(dec5_1)
        cat4 = torch.cat((unpool4,enc4_2),dim=1)
        # dim=1 -> channel 방향 concatenation
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.uppool3(dec4_1)
        cat3 = torch.cat((unpool3,enc3_2),dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        uppool2 = self.uppool2(dec3_1)
        cat2 = torch.cat((uppool2, enc2_2),dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        uppool1 = self.uppool1(dec2_1)
        cat1 = torch.cat((uppool1, enc1_2),dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        
        x = self.fc(dec1_1)

        return x

# dataloader 구현

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [i for i in lst_data if i.startswith('label')]
        lst_input = [i for i in lst_data if i.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)
    
    def __getitem__(self, index):
        label_ = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input_ = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label_ = label_/255.0
        input_ = input_/255.0

        if label_.ndim == 2:
            label_ = label_[:,:,np.newaxis]
        if input_.ndim == 2:
            input_ = input_[:,:,np.newaxis]

        data = {'input': input_, 'label': label_}

        if self.transform:
            data = self.transform(data)

        return data



class ToTensor(object):
    def __call__(self, data):
        label, inp = data['label'], data['input']

        label = label.transpose((2,0,1)).astype(np.float32)
        inp = inp.transpose((2,0,1)).astype(np.float32)
        
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(inp)} 

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
        
    def __call__(self,data):
        label, inp = data['label'], data['input']

        inp = (inp-self.mean) / self.std

        data = {'label': label, 'input': inp}

        return data

class RandomFlip(object):
    def __call__(self,data):
        label, inp = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            inp = np.fliplr(inp)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            inp = np.flipud(inp)

        data = {'label': label, 'input': inp}
        
        return data

# transforms.randomflip

# 네트워크 학습하기

transform = transforms.Compose([Normalization(), RandomFlip(), ToTensor()])

dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'),transform=transform)
# 데이터 255로 안나누고 즉, getitem제거후에 한번해볼것
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
# gpu로 학습시 num_worker 8로 해도됨.
# numworkers -> 멀티프로세싱으로 gpu의 활용도 높힘
dataset_val = Dataset(data_dir=os.path.join(data_dir,'val'), transform=transform)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

## 네트워크 생성하기

net = UNet().to(device)

## 손실함수 생성하기

fn_loss = nn.BCEWithLogitsLoss().to(device)

# optimizer 설정
optim = torch.optim.Adam(net.parameters(),lr=lr)

# 그밖 부수적인 variable 설정

num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

num_batch_train = np.ceil(num_data_train/batch_size)
# 20/4   만약 train이 30이라면 num_batxt_train == 8
num_batch_val = np.ceil(num_data_val/batch_size)
# ceil -> 올림시킴

# 그밖 부수적인 functions 설정
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
# Matplotlib로 시각화하기 위해서 채널변환
fn_denorm = lambda x, mean, std: (x*std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

#  SUMMARY WRITER 는 안됨

# 네트워크 학습시키기

st_epoch = 0
# net, optim, st_epoch = ckpt_dir=ckpt_dir, net=net, optim=optim


for epoch in range(st_epoch + 1, num_epoch + 1):
    net.train()
    loss_arr = []

    for batch, data in enumerate(loader_train, 1):
        # enumerate(array, start_index)
        # forward pass
        label = data['label'].to(device)
        input = data['input'].to(device)

        output = net(input)

        optim.zero_grad()

        loss = fn_loss(output, label)
        loss.backward()

        optim.step()

        loss_arr += [loss.item()]

        print("train: epoch %04d / %04d| batch %04d / %04d | loss %04f" %
                epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr))
        
        # label - fn_tonumpy(label)


def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save({'net':net.state_dict(), 'optim': optim.state_dict()},
                "./%s/model_epoch%d.pth" % (ckpt_dir, epoch))

def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch
    
    ckpt_lst = os.listdir(ckpt_dir)
    # ckpt_dir => './checkpoint'
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit,f))))
    # 이름내부의 숫자만 들고와짐 isdigit

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))
    # 맞나? 아닌가? 제일 큰 epoch만 들고오네 ㅋㅋ
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('pth')[0])
    # 이부분 이해잘안가네 ㄷ
    return net, optim, epoch




with torch.no_grad():
    net.eval()
    loss_arr = []

    for batch, data in enumerate(loader_val, 1):
        # forward pass
        label = data['label'].to(device)
        inp = data['input'].to(device)

        output = net(input)

        # 손실함수 계산하기
        loss = fn_loss(output, label)
        loss_arr += [loss.item()]

        print("valid: epoch %04d / %04d | batch %04d / %04d | loss %.4f " %
                    epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr))
                    # num_batch_val, num_batch_train 잘모르겠네,....
    save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)


