# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 18:58:01 2023

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
from evaluation import *
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from models_new import *
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
img = plt.imread("brodatz.png")
img = img[:,:,0]
img = resize(img,(128,128))
f = torch.tensor(img)
f = f 
f = f.unsqueeze(0).to(device)

args.output_directory = args.output_directory+"/" + args.dataset + "/patient_"+ str(args.patient).zfill(2) +  args.experiment
#define directory where you want to have your outputs saved
name = "/S2S_Method_"+ args.method + "_Lambda_" + str(args.lam) + "_ratio_"+ str(args.ratio) +'_lr_'+str(args.learning_rate)
path = args.output_directory+  name
args.lam=0.05

print(path)

def create_dir(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)
    print("Created Directory : ", dir)
  else:
    print("Directory already existed : ", dir)
  return dir
args.patient= 99
args.patient = 88
create_dir(path)
#data = np.load(
 #   'D:/DenoiSeg/DSB2018_n20.npz', allow_pickle=True)
data = np.load("C:/Users/johan/Desktop/hörnchen/joint_denoising_segmentation/data/"+args.dataset+"/train/train_data.npz")
f = torch.tensor(data["X_train"][args.patient:args.patient+1]).to(device)
f = f +0.1* torch.randn_like(f)*torch.max(f)
gt = data["Y_train"]
f_denoising = torch.clone(f)
args.fid = 0.001
mynet = Denseg_S2S(learning_rate = args.learning_rate, lam = args.lam, fid = args.fid)
mynet.initialize(f)
f=mynet.normalize(f)
f_norm = torch.mean(f**2)
mynet = Denseg_S2S(learning_rate = (1-f_norm)*args.learning_rate, lam = args.lam, fid = args.fid)
mynet.initialize(f)
f=mynet.normalize(f)
n_it = 50
args.method = "joint"
if args.method == "joint" or args.method == "cv":
    for i in range(n_it):
      if args.method == "joint":
        if i <1:
           for k in range(2000):
               mynet.segmentation_step(mynet.f)
        else:       
            for k in range(3000):
                mynet.segmentation_step2denoisers_acc(mynet.f)
               # mynet.segmentation_step2denoisers_acc_bg_constant(mynet.f)

        if i>0:
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
            plt.imshow(mynet.f2[0].cpu(),cmap ='inferno')
            plt.colorbar()    
            plt.subplot(3,2,5)
            plt.imshow(torch.abs(mynet.f1[0].cpu()-mynet.f[0].cpu()).cpu(),cmap ='inferno')
            plt.colorbar()   
            plt.subplot(3,2,6)
            plt.imshow(torch.abs(mynet.f2[0].cpu()-mynet.f[0].cpu()).cpu(),cmap ='inferno')
            plt.colorbar()   
            plt.show()
        ratio = 128**2/torch.sum(mynet.x)
        print(ratio)
        mynet.denoising_step_r1()
        mynet.denoising_step_r2()
        #mynet.denoising_step_constant()
          #  mynet.denoising_step_r2()

            #mynet.denoising_step_r1()
            #mynet.optimize_u()
            #mynet.combine_denoisings()

        #print(mynet.f.shape)
       # plt.imshow(mynet.f[0].cpu())

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
            plt.imshow(mynet.f2[0].cpu(),cmap ='inferno')
            plt.colorbar()    
            plt.subplot(3,2,5)
            plt.imshow(torch.abs(mynet.f1[0].cpu()-mynet.f[0].cpu()),cmap ='inferno')
            plt.colorbar()   
            plt.subplot(3,2,6)
            plt.imshow(torch.abs(mynet.f2[0].cpu()-mynet.f[0].cpu()),cmap ='inferno')
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
     
 



