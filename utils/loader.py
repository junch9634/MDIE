import argparse
import datetime
import logging
import math
import random
import time
import torch
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from skimage import io, transform
from PIL import Image
from torch.utils.data import DataLoader

def dir_name_change(name, idx):
    name_split = name.split('_')
    name_split[idx] = str(1)
    return '_'.join(name_split)


class NYUv2(object):
    def __init__(self, dir, mode = None, trans = None, gt_crop_size = None):
        HR_dir_name = 'gt/1_0_0_0_0'
        LR_dir_name = 'dataset'
        LR_dirs_name = 'multi_degrade_0'
        
        self.dir_HR = os.path.join(dir, HR_dir_name)
        self.dir_LR = os.path.join(dir, LR_dir_name, LR_dirs_name)
        
        self.mode = mode

        self.lis = sorted(os.listdir(self.dir_HR))

        self.transform = trans

        if gt_crop_size:
            self.gt_crop_size = int(gt_crop_size)
        else:
            self.gt_crop_size = gt_crop_size

        self.HR_list = []
        self.LR_list = []
        
        for img in self.lis:
            self.HR_list.append(os.path.join(self.dir_HR, img))
            self.LR_list.append(os.path.join(self.dir_LR, img))
            
        if mode == 'train':
            self.HR_list = self.HR_list[:int(len(self.HR_list)*4/5)]
            self.LR_list = self.LR_list[:int(len(self.LR_list)*4/5)]
        elif mode == 'val':
            self.HR_list = self.HR_list[int(len(self.HR_list)*4/5):]
            self.LR_list = self.LR_list[int(len(self.LR_list)*4/5):]
            

    def __len__(self):
        # return len(self.HRs_list)
        return len(self.HR_list)

    def __getitem__(self, idx):
        HR = self.HR_list[idx]
        LR = self.LR_list[idx]

        HR_img = np.asarray(Image.open(HR))
        LR_img = np.asarray(Image.open(LR))
        
        if self.mode == 'train' and self.gt_crop_size:
            h_gt, w_gt, _ = HR_img.shape
            h_lq, w_lq, _ = LR_img.shape

            top = random.randint(0, h_lq - int(self.gt_crop_size/4))
            left = random.randint(0, w_lq - int(self.gt_crop_size/4))

            HR_img = HR_img[top*4:top*4+int(self.gt_crop_size), left*4:left*4+int(self.gt_crop_size), :]
            LR_img = LR_img[top:top+int(self.gt_crop_size/4), left:left+int(self.gt_crop_size/4), :]
            
        if self.transform:
            HR_img = self.transform(HR_img.copy())
            LR_img = self.transform(LR_img.copy())

        im_name = HR

        return LR_img, HR_img, im_name



class NYUv2_sd(object):
    def __init__(self, dir, mode = None, trans = None, gt_crop_size = None):
        HR_dir_name = 'gt/1_0_0_0_0'
        LR_dir_name = 'pretrain_data'
        LR_dirs_name = '4_0_0_0_0'
        
        self.dir_HR = os.path.join(dir, HR_dir_name)
        self.dir_LR = os.path.join(dir, LR_dir_name, LR_dirs_name)
        
        # 1 = deblur, 2 = denoise, 3 = dehaze, 4 = derain
        self.dir_LR_deblur = os.path.join(dir, LR_dir_name, dir_name_change(LR_dirs_name, 1))
        self.dir_LR_denoise = os.path.join(dir, LR_dir_name, dir_name_change(LR_dirs_name, 2))
        self.dir_LR_dehaze = os.path.join(dir, LR_dir_name, dir_name_change(LR_dirs_name, 3))
        self.dir_LR_derain = os.path.join(dir, LR_dir_name, dir_name_change(LR_dirs_name, 4))
        
        self.mode = mode

        self.lis = sorted(os.listdir(self.dir_HR))

        self.transform = trans

        if gt_crop_size:
            self.gt_crop_size = int(gt_crop_size)
        else:
            self.gt_crop_size = gt_crop_size

        self.HR_list = []
        self.LR_list = []
        
        for img in self.lis:
            self.HR_list.append(os.path.join(self.dir_HR, img))
            self.LR_list.append(os.path.join(self.dir_LR, img))
            
        if mode == 'train':
            self.HR_list = self.HR_list[:int(len(self.HR_list)*4/5)]
            self.LR_list = self.LR_list[:int(len(self.LR_list)*4/5)]
        elif mode == 'val':
            self.HR_list = self.HR_list[int(len(self.HR_list)*4/5):]
            self.LR_list = self.LR_list[int(len(self.LR_list)*4/5):]
            

    def __len__(self):
        # return len(self.HRs_list)
        return len(self.HR_list)

    def __getitem__(self, idx):
        HR = self.HR_list[idx]
        LR = self.LR_list[idx]

        HR_img = np.asarray(Image.open(HR))
        LR_img = np.asarray(Image.open(LR))
        LR_deblur_img = np.asarray(Image.open(os.path.join(self.dir_LR_deblur, HR.split('/')[-1])))
        LR_denoise_img = np.asarray(Image.open(os.path.join(self.dir_LR_denoise, HR.split('/')[-1])))
        LR_dehaze_img = np.asarray(Image.open(os.path.join(self.dir_LR_dehaze, HR.split('/')[-1])))
        LR_derain_img = np.asarray(Image.open(os.path.join(self.dir_LR_derain, HR.split('/')[-1])))
        
        if self.mode == 'train' and self.gt_crop_size:
            h_gt, w_gt, _ = HR_img.shape
            h_lq, w_lq, _ = LR_img.shape

            top = random.randint(0, h_lq - int(self.gt_crop_size/4))
            left = random.randint(0, w_lq - int(self.gt_crop_size/4))

            HR_img = HR_img[top*4:top*4+int(self.gt_crop_size), left*4:left*4+int(self.gt_crop_size), :]
            LR_img = LR_img[top:top+int(self.gt_crop_size/4), left:left+int(self.gt_crop_size/4), :]
            LR_deblur_img = LR_deblur_img[top:top+int(self.gt_crop_size/4), left:left+int(self.gt_crop_size/4), :]
            LR_denoise_img = LR_denoise_img[top:top+int(self.gt_crop_size/4), left:left+int(self.gt_crop_size/4), :]
            LR_dehaze_img = LR_dehaze_img[top:top+int(self.gt_crop_size/4), left:left+int(self.gt_crop_size/4), :]
            LR_derain_img = LR_derain_img[top:top+int(self.gt_crop_size/4), left:left+int(self.gt_crop_size/4), :]
            
        if self.transform:
            HR_img = self.transform(HR_img.copy())
            LR_img = self.transform(LR_img.copy())
            LR_deblur_img = self.transform(LR_deblur_img.copy())
            LR_denoise_img = self.transform(LR_denoise_img.copy())
            LR_dehaze_img = self.transform(LR_dehaze_img.copy())
            LR_derain_img = self.transform(LR_derain_img.copy())
            
        
        single_distorted_input = [LR_deblur_img, LR_denoise_img, LR_dehaze_img, LR_derain_img]            
        im_name = HR

        return LR_img, single_distorted_input, HR_img, im_name



class REDS_Dataset_MULT(object):
    def __init__(self, dir, mode=None, trans = None, gt_crop_size = None):
        HR_dir_name = 'gt/1_0_0_0_0'
        LR_dir_name = 'dataset/multi_degrade'
        
        self.mode = mode
        self.dir_HR = os.path.join(dir, HR_dir_name)
        self.dir_LR = os.path.join(dir, LR_dir_name)
        
        # 1 = deblur, 2 = denoise, 3 = dehaze, 4 = derain
        self.lis = sorted(os.listdir(self.dir_HR))
        self.transform = trans

        if gt_crop_size:
            self.gt_crop_size = int(gt_crop_size)
        else:
            self.gt_crop_size = gt_crop_size

        self.HR_list = []
        self.LR_list = []
        
        for img in self.lis:
            self.HR_list.append(os.path.join(self.dir_HR, img))

        if mode == 'train':
            self.HR_list = self.HR_list[:int(len(self.HR_list)*4/5)]
        elif mode == 'val':
            self.HR_list = self.HR_list[int(len(self.HR_list)*4/5):]
            

    def __len__(self):
        return len(self.HR_list)

    def __getitem__(self, idx):
        HR = self.HR_list[idx]
        HR_img = np.asarray(Image.open(HR))

        img_index = HR.split('/')[-1].rstrip('.jpg')
        LR = os.path.join(self.dir_LR, '%s_0.jpg' %(img_index))
        LR_img = np.asarray(Image.open(LR))

        LR_deblur_img = np.asarray(Image.open(os.path.join(self.dir_LR, '%s_1.jpg' %(img_index) )))
        LR_denoise_img = np.asarray(Image.open(os.path.join(self.dir_LR, '%s_2.jpg' %(img_index) )))
        LR_dehaze_img = np.asarray(Image.open(os.path.join(self.dir_LR, '%s_3.jpg' %(img_index) )))
        LR_derain_img = np.asarray(Image.open(os.path.join(self.dir_LR, '%s_4.jpg' %(img_index) )))

        
        if self.mode == 'train' and self.gt_crop_size:
            h_gt, w_gt, _ = HR_img.shape
            h_lq, w_lq, _ = LR_img.shape

            top = random.randint(0, h_lq - int(self.gt_crop_size/4))
            left = random.randint(0, w_lq - int(self.gt_crop_size/4))

            HR_img = HR_img[top*4:top*4+int(self.gt_crop_size), left*4:left*4+int(self.gt_crop_size), :]
            LR_img = LR_img[top:top+int(self.gt_crop_size/4), left:left+int(self.gt_crop_size/4), :]
            LR_deblur_img = LR_deblur_img[top:top+int(self.gt_crop_size/4), left:left+int(self.gt_crop_size/4), :]
            LR_denoise_img = LR_denoise_img[top:top+int(self.gt_crop_size/4), left:left+int(self.gt_crop_size/4), :]
            LR_dehaze_img = LR_dehaze_img[top:top+int(self.gt_crop_size/4), left:left+int(self.gt_crop_size/4), :]
            LR_derain_img = LR_derain_img[top:top+int(self.gt_crop_size/4), left:left+int(self.gt_crop_size/4), :]
            
        if self.transform:
            HR_img = self.transform(HR_img.copy())
            LR_img = self.transform(LR_img.copy())
            LR_deblur_img = self.transform(LR_deblur_img.copy())
            LR_denoise_img = self.transform(LR_denoise_img.copy())
            LR_dehaze_img = self.transform(LR_dehaze_img.copy())
            LR_derain_img = self.transform(LR_derain_img.copy())
            
        
        single_distorted_input = [LR_deblur_img, LR_denoise_img, LR_dehaze_img, LR_derain_img]            
        im_name = HR
        return LR_img, single_distorted_input, HR_img, im_name