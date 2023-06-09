# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:52:26 2023

@author: johan
"""




import os
import numpy as np
from math import sqrt
from Functions_pytorch import *
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from skimage.color import rgb2gray
import scipy
from skimage.transform import resize
from skimage.color import rgb2gray
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model_N2F import *
from utils import *
from torch.optim import Adam
import time
import argparse
VERBOSE = 1
# ----

parser = argparse.ArgumentParser(description='Arguments for segmentation network.')
parser.add_argument('--output_directory', type=str, 
                    help='directory for outputs', default="C:/Users/johan/Desktop/hörnchen/joint_denoising_segmentation/data")
parser.add_argument('--input_directory', type=str, 
                    help='directory for input files', default = "C:/Users/johan/Desktop/hörnchen/joint_denoising_segmentation/data")
parser.add_argument('--learning_rate', type=float, 
                    help='learning rate', default=0.00001)
parser.add_argument('--method', type = str, help="joint/sequential or only Chan Vese cv", default = "joint")
parser.add_argument('--lam', type = float, help = "regularization parameter of CV", default = 0.0000001)
parser.add_argument('--ratio', type = int, help = "What is the ratio of masked pixels in N2v", default = 0.3)
parser.add_argument('--experiment', type = str, help = "What hyperparameter are we looking at here? Assigns the folder we want to produce with Lambda, if we make lambda tests for example", default = "/Lambda")
parser.add_argument('--patient', type = int, help = "Which patient index do we use", default = 0)
parser.add_argument('--dataset', type = str, help = "Which dataset", default = "DSB2018_n20")
parser.add_argument('--fid', type = float, help = "do we add fidelity term?", default = 0.0)


device = 'cuda:0'

args = parser.parse_args()
os.chdir("C:/Users/johan/Desktop/hörnchen/joint_denoising_segmentation")
# img = plt.imread("squirrel.jpg")
# img = img[:,:,1]
# img = resize(img,(128,128))
# f = torch.tensor(img)
# f = f 
# f = f.unsqueeze(0).to(device)

args.output_directory = args.output_directory+"/" + args.dataset + "/patient_"+ str(args.patient).zfill(2) +  args.experiment
#define directory where you want to have your outputs saved
name = "/S2S_Method_"+ args.method + "_Lambda_" + str(args.lam) + "_ratio_"+ str(args.ratio) +'_lr_'+str(args.learning_rate)
path = args.output_directory+  name
args.lam=0.075#75

print(path)

def create_dir(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)
    print("Created Directory : ", dir)
  else:
    print("Directory already existed : ", dir)
  return dir
args.patient= 14

create_dir(path)
#data = np.load(
 #   'D:/DenoiSeg/DSB2018_n20.npz', allow_pickle=True)
data = np.load("C:/Users/johan/Desktop/hörnchen/joint_denoising_segmentation/data/"+args.dataset+"/train/train_data.npz")
#
f = torch.tensor(data["X_train"][args.patient:args.patient+1]).to(device)
gt = torch.tensor(data["Y_train"][args.patient:args.patient+1]).to(device)
#f = torch.tensor(H).to(device).unsqueeze(0)

f = f #+0.2* torch.randn_like(f)*torch.max(f)
#gt = data["Y_train"]
f_denoising = torch.clone(f)
args.fid = 0.001
mynet = Denseg_S2S(learning_rate = args.learning_rate, lam = args.lam, fid = args.fid)
mynet.initialize(f)
f=mynet.normalize(f)
mynet = Denseg_S2S(learning_rate = args.learning_rate, lam = args.lam, fid = args.fid)
mynet.initialize(f)
f=mynet.normalize(f)
n_it =50
args.method = "joint"
cols = []
if args.method == "joint" or args.method == "cv":
    for i in range(n_it):
      if args.method == "joint":
        if i <1:
           mynet.x = (mynet.f>0.5).float()
           mynet.x_orig= torch.clone(mynet.x)
           for k in range(0):
               mynet.segmentation_step(mynet.f)
        else:       
            for k in range(5000):
                mynet.segmentation_step2denoisers_acc_bg_constant(mynet.f)
                #now we want to investigate the behaviour of the dice coefficient
                gt_bin = torch.clone(gt)
                gt_bin[gt_bin > 1] = 1
                seg = torch.round(mynet.x)
                fp =torch.sum(seg*(1-gt_bin))
                fn = torch.sum((1-seg)*gt_bin)
                tp = torch.sum(seg*gt_bin)
                tn = torch.sum((1-seg)*(1-gt_bin))
                dice = 2*tp/(2*tp + fn + fp)    
                mynet.Dice.append(((1-dice)*100).cpu())
            plt.subplot(2,2,1)
            plt.imshow((mynet.old_x + mynet.tau*div(mynet.p1) - mynet.tau*((mynet.fid1)**2) +  mynet.tau*((mynet.fid2**2))).cpu()[0])
            plt.colorbar()
            plt.subplot(2,2,2)
            plt.imshow((mynet.tau*div(mynet.p1).cpu()[0]))
            plt.colorbar()
            plt.subplot(2,2,3)
            plt.imshow((( - mynet.tau*((mynet.fid1)**2))).cpu()[0])
            plt.colorbar()
            plt.subplot(2,2,4)
            plt.imshow((( mynet.tau*((mynet.fid2**2))).cpu()[0]))
            plt.colorbar()
            
            plt.show()# proximity operator of indicator function on [0,1]
    
                ######################end of evaluation###############################
            # mynet.sigma_tv = mynet.sigma_tv*2**(i-1)
            #     #self.tau =  0.95/(4 + 2*5)
            # mynet.tau  = mynet.tau/2**(i-1)
            #plt.plot(mynet.en)
            #plt.plot(mynet.fid)
            plt.plot(mynet.fidelity_fg, label = "foreground_loss")
            plt.plot(mynet.fidelity_bg[:], label = "background_loss")
            plt.plot(mynet.tv, label = "TV")
            plt.plot(mynet.fidelity_fg_d_bg, label = "fg denoiser on bg")
            plt.plot(mynet.fidelity_bg_d_fg, label = "bg denoiser on fg")
            plt.plot(mynet.difference, label = "fg loss-bg loss on whole image")

            #plt.plot(mynet.en[:], label = "energy")
            #plt.plot(mynet.fid[:], label = "fidelity")
              #  plt.plot(self.lam*np.array(self.tv[498:]))
            plt.legend()
            plt.show()
            
            plt.plot(mynet.en,label = 'total energy')
            plt.plot(mynet.Dice,label = 'Dice')

            plt.legend()
            plt.show()
               # mynet.segmentation_step2denoisers_acc_bg_constant(mynet.f)

        if i>0:
            kernel = torch.ones(1,1,25,25)/625
            kernel = kernel
            diff1 = np.abs(mynet.f1[0].cpu()-mynet.f[0].cpu()).float().unsqueeze(0)
            diff2 = np.abs(mynet.mu_r2.cpu()-mynet.f[0].cpu()).float().unsqueeze(0)
           # sm_diff1 = torch.nn.functional.conv2d(diff1.unsqueeze(0).unsqueeze(0), kernel,padding = 12).squeeze(0)
            plt.subplot(3,2,1)
            plt.imshow(torch.round(mynet.x[0]).cpu())
            plt.colorbar()
            plt.subplot(3,2,2)
            plt.imshow(f[0].cpu())
            plt.colorbar()
            plt.subplot(3,2,3)
            plt.imshow(mynet.f1[0].cpu(),cmap ='inferno')
            plt.colorbar()
            plt.subplot(3,2,4)
            plt.imshow(mynet.f1[0].cpu(),cmap ='inferno')
            plt.colorbar()    
            plt.subplot(3,2,5)
            plt.imshow(diff1[0],cmap ='inferno')
            plt.colorbar()   
            plt.subplot(3,2,6)
            plt.imshow(diff1[0]>diff2[0],cmap ='inferno')
            plt.colorbar()   
            plt.title("Iteration"+str(i))
            plt.show()
        ratio = 128**2/torch.sum(mynet.x)
        plt.subplot(2,1,1)#
        plt.plot(mynet.val_loss_list_N2F[-1000:])
        plt.subplot(2,1,2)
        plt.plot(mynet.energy_denoising[-1000:])
        plt.show()
        #mynet.denoising_step_r1()
       # mynet.reinitialize_network()
        cols.append(2000*[0.0])
        if i>-1:
            mynet.N2Fstep()
            mynet.denoising_step_r2()
            # mynet.p = gradient(f)
            # mynet.x = torch.clone(mynet.f)
            # mynet.x_tilde = torch.clone(mynet.f)


        if i%1 ==0:
            plt.subplot(3,2,1)
            plt.imshow(mynet.x[0].cpu())
            plt.colorbar()
            plt.subplot(3,2,2)
            plt.imshow(f[0].cpu())
            plt.colorbar()
            plt.subplot(3,2,3)
            plt.imshow(mynet.f1[0].cpu(),cmap ='inferno')
            plt.colorbar()
            plt.subplot(3,2,4)
            plt.imshow(mynet.f1[0].cpu(),cmap ='inferno')
            plt.colorbar()    
            plt.subplot(3,2,5)
            plt.imshow(torch.abs(mynet.f1[0].cpu()-mynet.f[0].cpu()),cmap ='inferno')
            plt.colorbar()   
            plt.subplot(3,2,6)
            plt.imshow(torch.abs(mynet.f1[0].cpu()-mynet.f[0].cpu()),cmap ='inferno')
            plt.colorbar()   
            plt.show()

if args.method == "joint":
    mynet_sequential = Denseg_S2S(learning_rate = (1-f_norm)*args.learning_rate, lam = args.lam)
    mynet_sequential.initialize(mynet.f)

    for i in range(n_it):
        mynet_sequential.segmentation_step(mynet_sequential.f)
        
if args.method == "joint":        
    np.savez_compressed(path + "/results.npz", denoisings = mynet.f, segmentations_joint = mynet.x_tilde[0].cpu(), segmentations_sequential = mynet_sequential.x_tilde[0].cpu() , tv = mynet.tv)        

if args.method == "cv":
    np.savez_compressed(path + "/results.npz",  segmentations_cv = mynet.x_tilde[0].cpu() )        
     
 



