# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:53:00 2023

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
from torch.nn import init, ReflectionPad2d, ZeroPad2d
from torch.optim import lr_scheduler
from utils import *
from Functions_pytorch import *
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam



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
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size = 3, padding = 0)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size = 3, padding = 0)
        self.conv3 = nn.Conv2d(out_channel, final_channel, kernel_size = 3, padding = 0)
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
        learning_rate: float = 1e-3,
        lam: float = 0.01,
        ratio: float = 0.7,
        device: str = 'cuda:0',
        fid: float = 0.1,
        verbose = False,
        
    ):
        self.learning_rate = learning_rate
        self.lam = lam
        self.lam2 = 0.1
        self.ratio = ratio
        self.DenoisNet_r1 = self2self(1,0.3).to(device)
        self.DenoisNet_r2 = self2self(1,0.3).to(device)
        self.optimizer_r1 = Adam(self.DenoisNet_r1.parameters(),
                     lr=self.learning_rate, betas=(0.5, 0.999))
        self.optimizer_r2 = Adam(self.DenoisNet_r2.parameters(),
                     lr=self.learning_rate, betas=(0.5, 0.999))
        self.sigma_fid = 1.0/5
        self.sigma_tv = 1/np.sqrt(10+5)
        #self.tau =  0.95/(4 + 2*5)
        self.tau = 1/np.sqrt(10+5)
        self.tau_fid1 = self.tau/3
        self.theta = 1.0
        self.difference = []
        self.p = []
        self.q=[]
        self.r = []
        self.x_tilde = []
        self.device = device
        self.f_std = []
        self.Dice=[]
        self.Dice.append(1)
        self.fid=[]
        self.tv=[]
        self.tv_plot=[]
        self.fidelity_fg = []
        self.fidelity_bg = []
        self.en = []
        self.iteration = 0
        self.f1 = []
        self.f2 = []
        self.verbose = True
        self.Npred = 100
        self.denois_its = 1000
        self.loss_s2s=[]
        self.loss_list_N2F=[]
        self.net = Net()
        self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.energy_denoising = []
        self.val_loss_list_N2F = []
        self.bg_loss_list = []
        self.number_its_N2F=2000
        self.fidelity_fg_d_bg =[]
        self.fidelity_bg_d_fg = []
        self.old_x = 0
        self.fid1 = 0
        self.fid2= 0
        self.p1 = 0
        
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



##################### accelerated segmentation algorithm bg constant############################
    def segmentation_step2denoisers_acc_bg_constant(self,f):
        f_orig = torch.clone(f).to(self.device)
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
        # Update primal variables
        x_old = torch.clone(self.x)  
        # compute difference between noisy input and denoised image
        diff1 = (f_orig-f1).float()
        #compute difference between constant of background and originial noisy image
        diff2 = (f_orig - self.mu_r2)

        # constant difference term
        #filteing for smoother differences between denoised images and noisy input images
        self.x = proj_unitintervall(x_old + self.tau*div(p1) - self.tau*((diff1)**2) +  self.tau*((diff2**2))) # proximity operator of indicator function on [0,1]
        self.old_x = x_old
        self.fid1 = diff1
        self.fid2= diff2
        self.p1 = p1
        ######acceleration variables
        self.theta=1
        ###### 
        self.x_tilde = self.x + self.theta*(self.x-x_old)
        if self.verbose == True:
            fidelity = torch.sum((diff1)**2*self.x)+ torch.sum((diff2)**2*(1-self.x))
            fid_den = (torch.sum((diff1)**2*self.x)).cpu()
            fid_fg_denoiser_bg = (torch.sum((diff1)**2*(1-self.x))).cpu()
            fid_bg_denoiser_fg = (torch.sum((diff2)**2*(self.x))).cpu()
            self.fidelity_bg_d_fg.append(fid_bg_denoiser_fg)
            self.fidelity_fg_d_bg.append(fid_fg_denoiser_bg)
            self.fidelity_fg.append(fid_den)
            #self.difference.append(diff1-diff2)
            fid_const =( torch.sum((diff2**2*(1-self.x)))).cpu()
            self.fidelity_bg.append(fid_const)
            total = norm1(gradient(self.x))
            self.fid.append(fidelity.cpu())
            tv_p = norm1(gradient(self.x))
            tv_pf = norm1(gradient(self.x*f_orig))
            self.tv.append(total.cpu())
            energy = fidelity + self.lam*tv_p
            self.en.append(energy.cpu())
          #  plt.plot(np.array(self.tv), label = "TV")
            if self.iteration %5999 == 1:  
                plt.plot(np.array(self.fidelity_fg), label = "forground_loss")
                plt.plot(np.array(self.fidelity_bg[:]), label = "background_loss")
                plt.plot(np.array(self.en[:]), label = "energy")
                plt.plot(np.array(self.fid[:]), label = "fidelity")
                plt.plot(np.array(self.tv), label = "TV")

              #  plt.plot(self.lam*np.array(self.tv[498:]))
                plt.legend()
                plt.show()





    def denoising_step_r2(self):
        f = torch.clone(self.f_std[:,:,:,0])
     #   loss_mask = torch.clone(self.x<0.5).detach()
        #loss_mask = 1-self.x
        #self.mu_r2 = torch.mean(f[loss_mask])
        self.mu_r2 = torch.sum((f*(1-self.x))/torch.sum(1-self.x))
        

    
    def reinitialize_network(self):
        self.net = Net()
        self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def N2Fstep(self):

        print("learning_rate"+str(self.learning_rate))
        f = torch.clone(self.f_std[:,:,:,0])
        loss_mask = torch.clone(self.x>0.5).detach()        
       # loss_mask = self.x
        #mu_r1 = torch.mean(f[loss_mask])
        #print('mu_1 is', mu_r1)9
        img = f[0].cpu().numpy()* loss_mask[0].cpu().numpy()#*loss_mask[0].cpu().numpy()
        img = np.expand_dims(img,axis=0)
        img = np.expand_dims(img, axis=0)
        
        img_test = f[0].cpu().numpy()
        img_test = np.expand_dims(img_test,axis=0)
        img_test  = np.expand_dims(img_test, axis=0)
        
        minner = np.min(img)
        img = img -  minner
        maxer = np.max(img)
        img = img/ maxer
        img = img.astype(np.float32)
        img = img[0,0]
        
        minner_test = np.min(img_test)
        img_test = img_test -  minner_test
        maxer_test = np.max(img_test)
        img_test = img_test/ maxer
        img_test = img_test.astype(np.float32)
        img_test = img_test[0,0]

        shape = img.shape

         
        listimgH_mask = []
        listimgH = []
        Zshape = [shape[0],shape[1]]
        if shape[0] % 2 == 1:
            Zshape[0] -= 1
        if shape[1] % 2 == 1:
            Zshape[1] -=1  
        imgZ = img[:Zshape[0],:Zshape[1]]
        imgM = loss_mask[0,:Zshape[0],:Zshape[1]]
        
        imgin = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
        imgin2 = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
                 
        imgin_mask = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
        imgin2_mask = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
        for i in range(imgin.shape[0]):
            for j in range(imgin.shape[1]):
                if j % 2 == 0:
                    imgin[i,j] = imgZ[2*i+1,j]
                    imgin2[i,j] = imgZ[2*i,j]
                    imgin_mask[i,j] = imgM[2*i+1,j]
                    imgin2_mask[i,j] = imgM[2*i,j]
                if j % 2 == 1:
                    imgin[i,j] = imgZ[2*i,j]
                    imgin2[i,j] = imgZ[2*i+1,j]
                    imgin_mask[i,j] = imgM[2*i,j]
                    imgin2_mask[i,j] = imgM[2*i+1,j]
        imgin = torch.from_numpy(imgin)
        imgin = torch.unsqueeze(imgin,0)
        imgin = torch.unsqueeze(imgin,0)
        imgin = imgin.to(self.device)
        imgin2 = torch.from_numpy(imgin2)
        imgin2 = torch.unsqueeze(imgin2,0)
        imgin2 = torch.unsqueeze(imgin2,0)
        imgin2 = imgin2.to(self.device)
        listimgH.append(imgin)
        listimgH.append(imgin2)
        
        
        imgin_mask = torch.from_numpy(imgin_mask)
        imgin_mask = torch.unsqueeze(imgin_mask,0)
        imgin_mask = torch.unsqueeze(imgin_mask,0)
        imgin_mask = imgin_mask.to(self.device)
        imgin2_mask = torch.from_numpy(imgin2_mask)
        imgin2_mask = torch.unsqueeze(imgin2_mask,0)
        imgin2_mask = torch.unsqueeze(imgin2_mask,0)
        imgin2_mask = imgin2_mask.to(self.device)
        listimgH_mask.append(imgin_mask)
        listimgH_mask.append(imgin2_mask)        
         
        listimgV = []
        listimgV_mask=[]
        Zshape = [shape[0],shape[1]]
        if shape[0] % 2 == 1:
            Zshape[0] -= 1
        if shape[1] % 2 == 1:
             Zshape[1] -=1  
        imgZ = img[:Zshape[0],:Zshape[1]]
        imgM = loss_mask[0,:Zshape[0],:Zshape[1]]

         
        imgin3 = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        imgin4 = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        imgin3_mask = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        imgin4_mask = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        for i in range(imgin3.shape[0]):
            for j in range(imgin3.shape[1]):
                if i % 2 == 0:
                    imgin3[i,j] = imgZ[i,2*j+1]
                    imgin4[i,j] = imgZ[i, 2*j]
                    imgin3_mask[i,j] = imgM[i,2*j+1]
                    imgin4_mask[i,j] = imgM[i, 2*j]
                if i % 2 == 1:
                    imgin3[i,j] = imgZ[i,2*j]
                    imgin4[i,j] = imgZ[i,2*j+1]
                    imgin3_mask[i,j] = imgM[i,2*j]
                    imgin4_mask[i,j] = imgM[i,2*j+1]
        imgin3 = torch.from_numpy(imgin3)
        imgin3 = torch.unsqueeze(imgin3,0)
        imgin3 = torch.unsqueeze(imgin3,0)
        imgin3 = imgin3.to(self.device)
        imgin4 = torch.from_numpy(imgin4)
        imgin4 = torch.unsqueeze(imgin4,0)
        imgin4 = torch.unsqueeze(imgin4,0)
        imgin4 = imgin4.to(self.device)
        listimgV.append(imgin3)
        listimgV.append(imgin4)
        
        imgin3_mask = torch.from_numpy(imgin3_mask)
        imgin3_mask = torch.unsqueeze(imgin3_mask,0)
        imgin3_mask = torch.unsqueeze(imgin3_mask,0)
        imgin3_mask = imgin3_mask.to(self.device)
        imgin4_mask = torch.from_numpy(imgin4_mask)
        imgin4_mask = torch.unsqueeze(imgin4_mask,0)
        imgin4_mask = torch.unsqueeze(imgin4_mask,0)
        imgin4_mask = imgin4_mask.to(self.device)
        listimgV_mask.append(imgin3_mask)
        listimgV_mask.append(imgin4_mask)        
        

        img = torch.from_numpy(img)
     
        img = torch.unsqueeze(img,0)
        img = torch.unsqueeze(img,0)
        img = img.to(self.device)
         
        listimgV1 = [[listimgV[0],listimgV[1]]]
        listimgV2 = [[listimgV[1],listimgV[0]]]
        listimgH1 = [[listimgH[1],listimgH[0]]]
        listimgH2 = [[listimgH[0],listimgH[1]]]
        listimg = listimgH1+listimgH2+listimgV1+listimgV2
         
        listimgV1_mask = [[listimgV_mask[0],listimgV_mask[1]]]
        listimgV2_mask = [[listimgV_mask[1],listimgV_mask[0]]]
        listimgH1_mask = [[listimgH_mask[1],listimgH_mask[0]]]
        listimgH2_mask = [[listimgH_mask[0],listimgH_mask[1]]]
        listimg_mask = listimgH1_mask+listimgH2_mask+listimgV1_mask+listimgV2_mask
        #net = Net()
        #net.to(self.device)
        #criterion = torch.sum((output-y)**2*(1-mask)*loss_mask)/torch.sum((1-mask)*loss_mask)

        #criterion = nn.MSELoss()
        #optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
         
         
        
        img_test = torch.from_numpy(img_test)
        img_test = torch.unsqueeze(img_test,0)
        img_test = torch.unsqueeze(img_test,0)
        img_test = img_test.to(self.device)
        
        running_loss1=0.0
        running_loss2=0.0
        maxpsnr = -np.inf
        timesince = 0
        last10 = [0]*105
        last10psnr = [0]*105
        cleaned = 0
        
        while timesince < self.number_its_N2F:
            
            indx = np.random.randint(0,len(listimg))
            data = listimg[indx]
            data_mask = listimg_mask[indx]
            inputs = data[0]
            labello = data[1]
            loss_mask = data_mask[1]
            
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss1 = torch.sum((outputs-labello)**2*loss_mask) #+ 0.1*torch.sum((outputs -  torch.mean(outputs))**2)#/torch.sum(loss_mask)
            loss = loss1
            running_loss1+=loss1.item()
            self.loss_list_N2F.append(loss.detach().cpu())
            loss.backward()
            self.optimizer.step()
             
             
            running_loss1=0.0
            with torch.no_grad():
                last10.pop(0)
                last10.append(cleaned*maxer+minner)
                outputstest = self.net(img_test)

                self.en.append((torch.sum((outputstest[0]-img_test[0])**2*self.x) + torch.sum((img_test[0] - torch.sum(img_test[0]*(1-self.x))/torch.sum(1-self.x))**2*(1-self.x)) + self.lam*norm1(gradient(self.x))).cpu())
                self.Dice.append(self.Dice[-1])
                self.val_loss_list_N2F.append((torch.sum((outputstest[0]-img_test[0])**2*self.x)).cpu())
                self.bg_loss_list.append((torch.sum((img_test[0] - torch.sum(img_test[0]*(1-self.x))/torch.sum(1-self.x))**2*(1-self.x))/torch.sum(1-self.x)).cpu())
                cleaned = outputstest[0,0,:,:].cpu().detach().numpy()
                noisy = img_test.cpu().detach().numpy()


                ps = -np.mean((noisy-cleaned)**2)
                last10psnr.pop(0)
                last10psnr.append(ps)
                if ps > maxpsnr:
                    maxpsnr = ps
                    outclean = cleaned*maxer+minner
                    timesince = 0
                else:
                    timesince+=1.0
        plt.imshow(((outputstest[0].cpu()-img_test[0].cpu())**2*self.x.cpu())[0])
        plt.colorbar()
        plt.show()            
        H = np.mean(last10, axis=0)
        try: 
            running_loss = torch.mean(torch.stack(self.val_loss_list_N2F[-101:-1]))
            print(running_loss)
        except:
            running_loss = 1e10
        if self.val_loss_list_N2F[-1] > running_loss:
            for g in self.optimizer.param_groups:
                g['lr'] /= 3
            print('new learning rate is ', g['lr'])


        # if np.sum(np.round(H[1:-1,1:-1]-np.mean(H[1:-1,1:-1]))>0) <= 25 and self.learning_rate != 0.000005:
        #     self.learning_rate = 0.000005
        #     print("Reducing learning rate")
        # else:
        #     notdone = False

        #print(H.shape)
        self.f1 = torch.from_numpy(H).to(self.device)
        print(self.learning_rate)
        plt.plot(self.energy_denoising, label = "energy_denoising")
        plt.legend()
        plt.show()
        self.f1 = self.f1.unsqueeze(0)
        #self.number_its_N2F = 0#self.number_its_N2F*0.9

             
