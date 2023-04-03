# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 18:56:08 2023

@author: johan
"""




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import utils,models
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL
from tqdm import trange
from time import sleep
from scipy.io import loadmat
import torchvision.datasets as dset
from torch.utils.data import sampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from partialconv2d_S2S import PartialConv2d
    


from layer import *

import torch
import torch.nn as nn
from torch.nn import init, ReflectionPad2d, ZeroPad2d
from torch.optim import lr_scheduler
from utils import *
from Functions_pytorch import *
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam


class DIP(nn.Module):

    def __init__(self):
        super().__init__()  
        # define layers
        self.conv1 = nn.Conv2d(in_channels = 32, out_channels = 32 , kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32 , kernel_size = 3, padding = 1)
        self.down1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64 , kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64 , kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 128 , kernel_size = 3, padding = 1)
        self.conv6 = nn.Conv2d(in_channels = 128, out_channels = 128 , kernel_size = 3, padding = 1)
        self.conv7 = nn.Conv2d(in_channels = 128, out_channels = 256 , kernel_size = 3, padding = 1)
        self.conv8 = nn.Conv2d(in_channels = 256, out_channels = 128 , kernel_size = 3, padding = 1)
        self.conv9 = nn.Conv2d(in_channels = 128, out_channels = 128 , kernel_size = 3, padding = 1)
        self.conv10 = nn.Conv2d(in_channels = 128, out_channels = 64 , kernel_size = 3, padding = 1)
        self.conv11 = nn.Conv2d(in_channels = 64, out_channels = 64 , kernel_size = 3, padding = 1)
        self.conv12 = nn.Conv2d(in_channels = 64, out_channels = 32 , kernel_size = 3, padding = 1)
        self.conv13 = nn.Conv2d(in_channels = 32, out_channels = 32 , kernel_size = 3, padding = 1)
        self.conv14 = nn.Conv2d(in_channels = 32, out_channels = 32 , kernel_size = 3, padding = 1)
        self.conv15 = nn.Conv2d(in_channels = 32, out_channels = 1 , kernel_size = 3, padding = 1)
        self.actr = nn.ReLU()

    def forward(self, x):
        x = self.actr(self.conv1(x))
        x = self.actr(self.conv2(x))
        x = self.down1(x)
        x = self.actr(self.conv3(x))
        x = self.actr(self.conv4(x))
        x = self.down1(x)
        x = self.actr(self.conv5(x))
        x = self.actr(self.conv6(x))
        x = self.down1(x)
        x = self.actr(self.conv7(x))
        x = self.actr(self.conv8(x))
        x = torch.nn.functional.upsample(x,(32,32))
        x = self.actr(self.conv9(x))
        x = self.actr(self.conv10(x))
        x = torch.nn.functional.upsample(x,(64,64))
        x = self.actr(self.conv11(x))
        x = self.actr(self.conv12(x))
        x = torch.nn.functional.upsample(x,(128,128))
        x = self.actr(self.conv13(x))
        x = self.actr(self.conv14(x))
        x = self.conv15(x)

        return x
    



class EncodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, flag,padding = True,first_layer = False):
        super(EncodeBlock, self).__init__()
        if padding == True:
            self.conv = PartialConv2d(in_channel, out_channel, kernel_size = 3, padding = 0,bias=False)
        else:
            self.conv = PartialConv2d(in_channel, out_channel, kernel_size = 3, padding = 0,bias=False)
        #self.conv = nn.Conv2d(in_channel, out_channel, kernel_size = 3, padding = 1)
        self.nonlinear = nn.LeakyReLU(0.1)
        self.MaxPool = nn.MaxPool2d(2)
        self.flag = flag
        self.padding = padding
        self.pad_im = ReflectionPad2d(1)
        self.pad_mask = ZeroPad2d(1)
        self.first_layer = first_layer
        
    def forward(self, x, mask_in):
        if self.padding == True:
            x = self.pad_im(x)
            mask_in = self.pad_mask(mask_in)
            if self.first_layer == True:
                x = x*mask_in

        out1, mask_out = self.conv(x, mask_in = mask_in)
        out2 = self.nonlinear(out1)
        if self.flag:
            out = self.MaxPool(out2)
            mask_out = self.MaxPool(mask_out)
        else:
            out = out2
        return out, mask_out


class DecodeBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, final_channel = 3, p = 0.7, flag = False):
        super(DecodeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size = 3, padding = 0,bias=False)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size = 3, padding = 0,bias=False)
        self.conv3 = nn.Conv2d(out_channel, final_channel, kernel_size = 3, padding = 0,bias=False)
        self.nonlinear1 = nn.LeakyReLU(0.1)
        self.nonlinear2 = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        self.flag = flag
        self.Dropout = nn.Dropout(p)
        self.pad_im = ReflectionPad2d(1)
        
    def forward(self, x):
        x = self.pad_im(x)
        out1 = self.conv1(self.Dropout(x))
        out2 = self.nonlinear1(out1)
        out2 = self.pad_im(out2)
        out3 = self.conv2(self.Dropout(out2))
        out4 = self.nonlinear2(out3)
        if self.flag:
            out4 = self.pad_im(out4)
            out5 = self.conv3(self.Dropout(out4))
            out = self.sigmoid(out5)
        else:
            out = out4
        return out

class self2self(nn.Module):
    def __init__(self, in_channel, p):
        super(self2self, self).__init__()
        self.EB0 = EncodeBlock(in_channel, 48, flag=False,first_layer = False)
        self.EB1 = EncodeBlock(48, 48, flag=True)
        self.EB2 = EncodeBlock(48, 48, flag=True)
        self.EB3 = EncodeBlock(48, 48, flag=True)
        self.EB4 = EncodeBlock(48, 48, flag=True)
        #self.EB5 = EncodeBlock(48, 48, flag=True)
        self.EB6 = EncodeBlock(48, 48, flag=False)
        
        #self.DB1 = DecodeBlock(96, 96, 96,p=p)
        self.DB2 = DecodeBlock(96, 96, 96,p=p)
        self.DB3 = DecodeBlock(144, 96, 96,p=p)
        self.DB4 = DecodeBlock(144, 96, 96,p=p)
        self.DB5 = DecodeBlock(96+in_channel, 64, 32, in_channel,p=p, flag=True)
        
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        
        self.concat_dim = 1
    def forward(self, x, mask):
        out_EB0, mask = self.EB0(x, mask)
        out_EB1, mask = self.EB1(out_EB0, mask)
        out_EB2, mask = self.EB2(out_EB1, mask_in = mask)
        out_EB3, mask = self.EB3(out_EB2, mask_in = mask)
        out_EB4, mask = self.EB4(out_EB3, mask_in = mask)
        #out_EB5, mask = self.EB5(out_EB4, mask_in = mask)
        out_EB6, mask = self.EB6(out_EB4, mask_in = mask)
        
        out_EB6_up = self.Upsample(out_EB6)
        in_DB1 = torch.cat((out_EB6_up, out_EB3),self.concat_dim)
        #in_DB1 = torch.cat((out_EB6_up, out_EB4),self.concat_dim)
        #out_DB1 = self.DB1((in_DB1))
        
        #out_DB1_up = self.Upsample(out_DB1)
        #in_DB2 = torch.cat((out_DB1_up, out_EB3),self.concat_dim)
        out_DB2 = self.DB2((in_DB1))
        
        out_DB2_up = self.Upsample(out_DB2)
        in_DB3 = torch.cat((out_DB2_up, out_EB2),self.concat_dim)
        out_DB3 = self.DB3((in_DB3))
        
        out_DB3_up = self.Upsample(out_DB3)
        in_DB4 = torch.cat((out_DB3_up, out_EB1),self.concat_dim)
        out_DB4 = self.DB4((in_DB4))
        
        out_DB4_up = self.Upsample(out_DB4)
        in_DB5 = torch.cat((out_DB4_up, x),self.concat_dim)
        out_DB5 = self.DB5(in_DB5)
        
        return out_DB5 
    


#for noise to fast
class TwoCon(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = TwoCon(1, 64)
        self.conv2 = TwoCon(64, 64)
        self.conv3 = TwoCon(64, 64)
        self.conv4 = TwoCon(64, 64)  
        self.conv6 = nn.Conv2d(64,1,1)
        

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x = self.conv4(x3)
        x = torch.sigmoid(self.conv6(x))
        return x

# U-Net: Convolutional Networks for Biomedical Image Segmentation
# https://arxiv.org/abs/1505.04597
class UNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(UNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        self.pad = ReflectionPad2d(1)

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        """
        Encoder part
        """
        self.enc1_1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=3,padding = 0, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc1_2 = CNR2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.pool1 = Pooling2d(pool=2, type='avg')

        self.enc2_1 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc2_2 = CNR2d(2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.pool2 = Pooling2d(pool=2, type='avg')

        self.enc3_1 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc3_2 = CNR2d(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.pool3 = Pooling2d(pool=2, type='avg')

        self.enc4_1 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc4_2 = CNR2d(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.pool4 = Pooling2d(pool=2, type='avg')

        self.enc5_1 = CNR2d(8 * self.nch_ker, 2 * 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        """
        Decoder part
        """

        self.dec5_1 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.unpool4 = UnPooling2d(pool=2, type='nearest')

        self.dec4_2 = DECNR2d(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec4_1 = DECNR2d(8 * self.nch_ker,     4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.unpool3 = UnPooling2d(pool=2, type='nearest')

        self.dec3_2 = DECNR2d(2 * 4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec3_1 = DECNR2d(4 * self.nch_ker,     2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.unpool2 = UnPooling2d(pool=2, type='nearest')

        self.dec2_2 = DECNR2d(2 * 2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec2_1 = DECNR2d(2 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.unpool1 = UnPooling2d(pool=2, type='nearest')

        self.dec1_2 = DECNR2d(2 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec1_1 = DECNR2d(1 * self.nch_ker,     1 * self.nch_out, kernel_size=3, stride=1, norm=[],        relu=[],  drop=[], bias=False)

    def forward(self, x):

        """
        Encoder part
        """
        enc0 = self.pad(x)
        enc1 = self.enc1_2(self.enc1_1(enc0))
        pool1 = self.pool1(enc1)

        enc2 = self.enc2_2(self.enc2_1(pool1))
        pool2 = self.pool2(enc2)

        enc3 = self.enc3_2(self.enc3_1(pool2))
        pool3 = self.pool3(enc3)

        enc4 = self.enc4_2(self.enc4_1(pool3))
        pool4 = self.pool4(enc4)

        enc5 = self.enc5_1(pool4)

        """
        Encoder part
        """
        dec5 = self.dec5_1(enc5)

        unpool4 = self.unpool4(dec5)
        cat4 = torch.cat([enc4, unpool4], dim=1)
        dec4 = self.dec4_1(self.dec4_2(cat4))

        unpool3 = self.unpool3(dec4)
        cat3 = torch.cat([enc3, unpool3], dim=1)
        dec3 = self.dec3_1(self.dec3_2(cat3))

        unpool2 = self.unpool2(dec3)
        cat2 = torch.cat([enc2, unpool2], dim=1)
        dec2 = self.dec2_1(self.dec2_2(cat2))

        unpool1 = self.unpool1(dec2)
        cat1 = torch.cat([enc1, unpool1], dim=1)
        dec1 = self.dec1_1(self.dec1_2(cat1))

        x = dec1

        return x

# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
# https://arxiv.org/abs/1609.04802
class ResNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm', nblk=16):
        super(ResNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm
        self.nblk = nblk

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.enc1 = CNR2d(self.nch_in, self.nch_ker, kernel_size=3, stride=1, padding=1, norm=[], relu=0.0)

        res = []
        for i in range(self.nblk):
            res += [ResBlock(self.nch_ker, self.nch_ker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0, padding_mode='reflection')]
        self.res = nn.Sequential(*res)

        self.dec1 = CNR2d(self.nch_ker, self.nch_ker, kernel_size=3, stride=1, padding=1, norm=norm, relu=[])

        self.conv1 = Conv2d(self.nch_ker, self.nch_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.enc1(x)
        x0 = x

        x = self.res(x)

        x = self.dec1(x)
        x = x + x0

        x = self.conv1(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, nch_in, nch_ker=64, norm='bnorm'):
        super(Discriminator, self).__init__()

        self.nch_in = nch_in
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        # dsc1 : 256 x 256 x 3 -> 128 x 128 x 64
        # dsc2 : 128 x 128 x 64 -> 64 x 64 x 128
        # dsc3 : 64 x 64 x 128 -> 32 x 32 x 256
        # dsc4 : 32 x 32 x 256 -> 16 x 16 x 512
        # dsc5 : 16 x 16 x 512 -> 16 x 16 x 1

        self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.2)
        self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[],        relu=[], bias=False)

        # self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
        # self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=1, padding=1, norm=[], relu=0.2)
        # self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[], relu=[], bias=False)

    def forward(self, x):

        x = self.dsc1(x)
        x = self.dsc2(x)
        x = self.dsc3(x)
        x = self.dsc4(x)
        x = self.dsc5(x)

        # x = torch.sigmoid(x)

        return x


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def image_loader(image, device, p1, p2):
    """load image, returns cuda tensor"""
    loader = T.Compose([T.ToPILImage(),T.RandomHorizontalFlip(torch.round(torch.tensor(p1))),T.RandomVerticalFlip(torch.round(torch.tensor(p2))),T.ToTensor()])
    image = torch.tensor(image).float()
    image= loader(image)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.to(device)
class Denseg_S2S:
    def __init__(
        self,
        learning_rate: float = 4e-3,
        lam: float = 0.01,
        ratio: float = 0.7,
        device: str = 'cuda:0',
        fid: float = 0.1,
        verbose = False,
        
    ):
        self.learning_rate = learning_rate
        self.lam = lam
        self.ratio = ratio
        self.DenoisNet_r1 = self2self(1,0.3).to(device)
        self.DenoisNet_r2 = self2self(1,0.3).to(device)
        self.optimizer_r1 = Adam(self.DenoisNet_r1.parameters(),
                     lr=self.learning_rate, betas=(0.5, 0.999))
        self.optimizer_r2 = Adam(self.DenoisNet_r2.parameters(),
                     lr=self.learning_rate, betas=(0.5, 0.999))
        self.sigma_fid = 1.0/5
        self.sigma_tv = 1/2
        self.tau =  0.95/(4 + 2*5)
        self.theta = 1.0
        self.p = []
        self.q=[]
        self.r = []
        self.x_tilde = []
        self.device = device
        self.f_std = []
        self.fid=[]
        self.tv=[]
        self.tv_plot=[]
        self.en = []
        self.iteration = 0
        self.f1 = []
        self.f2 = []
        self.verbose = verbose
        self.Npred = 100
        self.denois_its = 2000
        
    def normalize(self,f):
        f = torch.tensor(f).float()
        f = (f-torch.min(f))/(torch.max(f)-torch.min(f))
        return f
    def standardise(self,f):
        f = torch.tensor(f).unsqueeze(3).float()
        f = (f - torch.mean(f))/torch.std(f)
        return f
        
    def initialize(self,f):
        #prepare input for denoiser
        f_train = torch.tensor(f).unsqueeze(3).float()
  #      f_train = (f_train - torch.mean(f_train))/torch.std(f_train)
        dataset = TensorDataset((f_train[:, :, :]), (f_train[:, :, :]))
        self.train_loader = DataLoader(dataset, batch_size=1, pin_memory=False)
        
        f_val = torch.clone(f_train)
        dataset_val = TensorDataset(f_val, f_val)
        self.val_loader = DataLoader(dataset_val, batch_size=1, pin_memory=False)
        self.f_std = torch.clone(f_train)


        #prepare input for segmentation
        f = self.normalize(f)
        #f = torch.rand_like(f)
        self.p = gradient(f)
        self.q = f
        self.r = f
        self.x_tilde = f
        self.x = f
        self.f = torch.clone(f)

 ######################## Classical Chan Vese step############################################       
    def segmentation_step(self,f):
        f_orig = torch.clone(f)
        
        # for segmentaion process, the input should be normalized and the values should lie in [0,1]
        
        '''-------------now the segmentation process starts-------------------'''
        ''' Update dual variables of image f'''
        p1 = proj_l1_grad(self.p + self.sigma_tv*gradient(self.x_tilde), self.lam)  # update of TV
        q1 = torch.ones_like(f)
        r1 = torch.ones_like(f)

        self.p = p1.clone()
        self.q = q1.clone()
        self.r = r1.clone()
        # Update primal variables
        x_old = torch.clone(self.x)  
        self.x = proj_unitintervall(x_old + self.tau*div(p1) - self.tau*adjoint_der_Fid1(x_old, f, self.q) - self.tau *
                               adjoint_der_Fid2(x_old, f, self.r))  # proximity operator of indicator function on [0,1]
        self.x_tilde = self.x + self.theta*(self.x-x_old)
        if self.verbose == True:
            fidelity = norm1(Fid1(self.x, f)) + norm1(Fid2(self.x,f))
            total = norm1(gradient(self.x))
            self.fid.append(fidelity.cpu())
            tv_p = norm1(gradient(self.x))
            self.tv.append(total.cpu())
            energy = fidelity +self.lam* total
            self.en.append(energy.cpu())
          #  plt.plot(np.array(self.tv), label = "TV")
            if self.iteration %2999 == 0:  
                plt.plot(np.array(self.en[:]), label = "energy")
                plt.plot(np.array(self.fid[:]), label = "fidelity")
              #  plt.plot(self.lam*np.array(self.tv[498:]))
                plt.legend()
                plt.show()
        self.iteration += 1


    def segmentation_step2denoisers_acc(self,f):
        f_orig = torch.clone(f).to(self.device)

        kernel = torch.ones(1,1,9,9)/81
        kernel = kernel.to(self.device)
        # for segmentaion process, the input should be normalized and the values should lie in [0,1]
        
        '''-------------now the segmentation process starts-------------------'''
        ''' Update dual variables of image f'''
 
        self.mu = 0.5
        p1 = proj_l1_grad(self.p + self.sigma_tv*gradient(self.x_tilde), self.lam)  # update of TV
        q1 = torch.ones_like(f)
        r1 = torch.ones_like(f)
        # Fidelity term without norm (change this in funciton.py)

        self.p = p1.clone()
        self.q = q1.clone()
        self.r = r1.clone()
        f1 = torch.clone(self.f1)
        f2 = torch.clone(self.f2)
        # Update primal variables
        x_old = torch.clone(self.x)  
        # compute difference between noisy input and denoised image
        diff1 = (f_orig-f1).float().unsqueeze(0)
        diff2 = (f_orig-f2).float().unsqueeze(0)
        #filteing for smoother differences between denoised images and noisy input images
        sm_diff1 = torch.nn.functional.conv2d(diff1, kernel,padding = 4).squeeze(0)
        sm_diff2 = torch.nn.functional.conv2d(diff2, kernel,padding = 4).squeeze(0)
        self.x = proj_unitintervall(((x_old + self.tau*div(p1) - self.tau*(sm_diff1)**2 + self.tau*(sm_diff2)**2 + self.tau*self.mu*self.x - self.tau*((sm_diff1)**2 - (sm_diff2)**2)) )/(1 + self.tau  * self.mu)) # proximity operator of indicator function on [0,1]
        ######acceleration variables
        self.theta = 1/np.sqrt(1 +2*self.tau)
        self.tau = self.theta*self.tau
        self.sigma_tv = self.sigma_tv/self.theta
    
        self.x_tilde = self.x + self.theta*(self.x-x_old)
        if self.verbose == True:
            fidelity = torch.mean((f_orig-f1)**2*self.x) + torch.mean((f_orig-f2)**2*(1-self.x))
            total = norm1(gradient(self.x))
            self.fid.append(fidelity.cpu())
            tv_p = norm1(gradient(self.x))
            self.tv.append(total.cpu())
            self.en.append(energy.cpu())
          #  plt.plot(np.array(self.tv), label = "TV")
            if self.iteration %2999 == 0:  
                plt.plot(np.array(self.en[:]), label = "energy")
                plt.plot(np.array(self.fid[:]), label = "fidelity")
              #  plt.plot(self.lam*np.array(self.tv[498:]))
                plt.legend()
                plt.show()

    def segmentation_step2denoisers(self,f):
        f_orig = torch.clone(f).to(self.device)

        kernel = torch.ones(1,1,9,9)/81
        kernel = kernel.to(self.device)
        # for segmentaion process, the input should be normalized and the values should lie in [0,1]
        
        '''-------------now the segmentation process starts-------------------'''
        ''' Update dual variables of image f'''
        p1 = proj_l1_grad(self.p + self.sigma_tv*gradient(self.x_tilde), self.lam)  # update of TV
        q1 = torch.ones_like(f)
        r1 = torch.ones_like(f)
        # Fidelity term without norm (change this in funciton.py)

        self.p = p1.clone()
        self.q = q1.clone()
        self.r = r1.clone()
        f1 = torch.clone(self.f1)
        f2 = torch.clone(self.f2)
        # Update primal variables
        x_old = torch.clone(self.x)  
        diff1 = (f_orig-f1).float().unsqueeze(0)
        diff2 = (f_orig-f2).float().unsqueeze(0)
        
        sm_diff1 = torch.nn.functional.conv2d(diff1, kernel,padding = 4).squeeze(0)
        sm_diff2 = torch.nn.functional.conv2d(diff2, kernel,padding = 4).squeeze(0)
        # plt.imshow((sm_diff1[0]**2).detach().cpu())
        # plt.colorbar()
        # plt.show()
        # plt.imshow((sm_diff2[0]**2).detach().cpu())
        # plt.colorbar()
        # plt.show()
        self.x = proj_unitintervall(x_old + self.tau*div(p1) - self.tau*(sm_diff1)**2 + self.tau*(sm_diff2)**2)  # proximity operator of indicator function on [0,1]
        self.x_tilde = self.x + self.theta*(self.x-x_old)
        if self.verbose == True:
            fidelity = torch.mean((f_orig-f1)**2*self.x) +torch.mean((f_orig-f2)**2*(1-self.x))
            total = norm1(gradient(self.x))
            self.fid.append(fidelity.cpu())
            tv_p = norm1(gradient(self.x))
            self.tv.append(total.cpu())
            self.en.append(energy.cpu())
          #  plt.plot(np.array(self.tv), label = "TV")
            if self.iteration %2999 == 0:  
                plt.plot(np.array(self.en[:]), label = "energy")
                plt.plot(np.array(self.fid[:]), label = "fidelity")
              #  plt.plot(self.lam*np.array(self.tv[498:]))
                plt.legend()
                plt.show()
        self.iteration += 1


    def denoising_step_r1(self):
        f = torch.clone(self.f_std[:,:,:,0])
        loss_mask = torch.clone(self.x>=0.75).detach()
        mu_r1 = torch.mean(f[loss_mask])
        ratio = (f.shape[-1]*f.shape[-2]/2)/torch.sum(loss_mask)
        img_input = torch.clone(f)
        w,h,c = f.shape
        p=self.ratio

        number_of_its=round(ratio.cpu().numpy()*self.denois_its)
        #slice_avg = torch.tensor([1,1,128,128]).to(device)
        for itr in range(number_of_its):
            self.optimizer_r1.zero_grad()
            p_mtx = np.random.uniform(size=[img_input.shape[0],img_input.shape[1],img_input.shape[2]])
            mask = (p_mtx>p).astype(np.double)*0.7
            y = torch.clone(img_input)
            p1 = np.random.uniform(size=1)
            p2 = np.random.uniform(size=1)
            img_input_tensor = image_loader(img_input, self.device, p1, p2)
            y = image_loader(y, self.device, p1, p2)
            mask = np.expand_dims(mask,0)            
            mask = torch.tensor(mask).to(self.device, dtype=torch.float32)

            self.DenoisNet_r1.train()
            img_input_tensor = img_input_tensor*mask*loss_mask
            output = self.DenoisNet_r1(img_input_tensor, mask*loss_mask)	    
            #loader = T.Compose([T.RandomHorizontalFlip(torch.round(torch.tensor(p1))),T.RandomVerticalFlip(torch.round(torch.tensor(p2)))])

            loss = torch.sum((output-y)**2*(1-mask)*loss_mask/torch.sum((1-mask)*loss_mask))+10*torch.mean((output-mu_r1)**2) 
            loss.backward()
            self.optimizer_r1.step()

            if (itr+1) == number_of_its:
                self.DenoisNet_r1.eval()
                sum_preds = np.zeros((img_input.shape[0],img_input.shape[1],img_input.shape[2]))
                for j in range(self.Npred):
                #generate bernoulli sampled instances
                    p_mtx_vl = np.random.uniform(size=img_input.shape)
                    mask_val = (p_mtx_vl>p).astype(np.double)*0.7
                    img_input_val = torch.clone(img_input)*torch.tensor(mask_val).to(self.device)
                    img_input_tensor_val = image_loader(img_input_val, self.device,0.0,0.0)
                    y_val = image_loader(torch.clone(img_input_val), self.device ,0.0,0.0)

                    mask_val = torch.tensor(mask_val).to(self.device, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        output_test = self.DenoisNet_r1(img_input_tensor_val,mask_val)
                        sum_preds[:,:,:] += output_test.detach().cpu().numpy()[0]

                avg_preds = np.squeeze(sum_preds)
                output = (torch.tensor(avg_preds))/self.Npred
        print(output.shape)
        self.f1=torch.clone(output.unsqueeze(0)).detach().to(self.device)
        print(self.f1.shape)

 

    def denoising_step_r2(self):
        f = torch.clone(self.f_std[:,:,:,0])
        loss_mask = torch.clone(self.x<0.25).detach()
        mu_r2 = torch.mean(f[loss_mask])
        img_input = torch.clone(f)
        w,h,c = f.shape
        p=self.ratio

        for itr in range(self.denois_its):
            self.optimizer_r2.zero_grad()
            p_mtx = np.random.uniform(size=[img_input.shape[0],img_input.shape[1],img_input.shape[2]])
            mask = (p_mtx>p).astype(np.double)*0.7
            y = torch.clone(img_input)
            p1 = np.random.uniform(size=1)
            p2 = np.random.uniform(size=1)
            img_input_tensor = image_loader(img_input, self.device, p1, p2)
            y = image_loader(y, self.device, p1, p2)
            mask = np.expand_dims(mask,0)            
            mask = torch.tensor(mask).to(self.device, dtype=torch.float32)

            self.DenoisNet_r2.train()
            img_input_tensor = img_input_tensor*mask*loss_mask
            output = self.DenoisNet_r2(img_input_tensor, mask*loss_mask)	    
            #loader = T.Compose([T.RandomHorizontalFlip(torch.round(torch.tensor(p1))),T.RandomVerticalFlip(torch.round(torch.tensor(p2)))])
            
            loss = torch.sum((output-y)**2*(1-mask)*loss_mask)/torch.sum((1-mask)*loss_mask)+15*torch.mean((output-mu_r2)**2) 
            loss.backward()
            self.optimizer_r2.step()
            
            if (itr+1) == self.denois_its:
	     
                self.DenoisNet_r2.eval()
                sum_preds = np.zeros((img_input.shape[0],img_input.shape[1],img_input.shape[2]))
                for j in range(self.Npred):
                #generate bernoulli sampled instances
                    p_mtx_vl = np.random.uniform(size=img_input.shape)
                    mask_val = (p_mtx_vl>p).astype(np.double)*0.7
                    img_input_val = torch.clone(img_input)*torch.tensor(mask_val).to(self.device)
                    img_input_tensor_val = image_loader(img_input_val, self.device,0.0,0.0)
                    y_val = image_loader(torch.clone(img_input_val), self.device ,0.0,0.0)
                    mask_val = torch.tensor(mask_val).to(self.device, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        output_test = self.DenoisNet_r2(img_input_tensor_val,mask_val)
                        sum_preds[:,:,:] += output_test.detach().cpu().numpy()[0]
                      
                avg_preds = np.squeeze(sum_preds)
                output = (torch.tensor(avg_preds))/self.Npred     
        
        print(output.shape)
        self.f2=torch.clone(output.unsqueeze(0).to(self.device)).detach()
        print(self.f2.shape)
        
    def combine_denoisings(self):
        #self.f[self.x>=0.5]=self.f1[self.x>=0.5].float()
        #self.f[self.x<0.5]=self.f2[self.x<0.5].float()
        self.f = self.x*self.f1+self.f2 * (1-self.x)

    
 