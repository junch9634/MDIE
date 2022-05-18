import os, argparse, glob, random, h5py
import numpy as np
import cv2
import albumentations as A
from tqdm import tqdm
from LinearMotionBlur import LinearMotionBlur
from tools import *

# settings
parser = argparse.ArgumentParser(description="data_generation")
parser.add_argument("--rain", action="store_true", help="Add Rain? (default: none)")
parser.add_argument("--haze", action="store_true", help="Add haze? (default: none)")
parser.add_argument("--downscale", action="store_true", help="Add downscale? (default: none)")
parser.add_argument("--blur", action="store_true", help="Add blur? (default: none)")
parser.add_argument("--noise", action="store_true", help="Add noise? (default: none)")
parser.add_argument("--beta", type=float, default=1.0, help="beta of haze[0.4 ~ 2.4]")
parser.add_argument("--a", type=float, default=0.5, help="a of haze[0.5 ~ 1.0]")
parser.add_argument("--scale", type=int, default=1, help="scale of downscale")
parser.add_argument("--kernelsize", type=int, default=5, help="kernelsize of blur[3, 5, 7]")
parser.add_argument("--angle", type=int, default=random.randint(0, 180), help="angle of blur")
parser.add_argument("--noisemode", type=str, default='poisson', help="gaussian or poisson")
parser.add_argument("--var", type=int, default=0.0001, help="var of noise[0.005, 0.010, 0.015, 0.020, 0.025]")


def main():
    global opt
    opt = parser.parse_args()

    nyu_depth = h5py.File('./nyu_depth_v2_labeled.mat', 'r')
    images = nyu_depth['images']
    depths = nyu_depth['depths']

    root = './data'

    dir_name = make_dirname(opt)    # base_data = '1_0_0_0_0'

    # if os.path.isdir(os.path.join(root, dir_name)):
    #     raise Exception('Already there')
    # else:
    #     os.mkdir(os.path.join(root, dir_name))

    for index in tqdm(range(len(images))):
        # if aaa >= len(glob.glob(os.path.join(root, dir_name+'/*'))):
        image = (images[index, :, :, :]).astype(float)
        image = np.swapaxes(image, 0, 2)
        image = image / 255
        image = image.astype('float32')

        depth = depths[index, :, :]
        maxhazy = depth.max()
        depth = (depth) / (maxhazy)
        depth = np.swapaxes(depth, 0, 1)
        depth = depth.astype('float32')

        image = degradation(image, depth, opt) * 255

        cv2.imwrite(os.path.join(root, dir_name, f'{index}.jpg'), image)


def make_dirname(opt):
    if opt.downscale:
        a = 4
    else:
        a = 1
    if opt.blur:
        b = opt.kernelsize
    else:
        b = 0
    if opt.noise:
        c = f'{opt.noisemode},{opt.var}'
    else:
        c = 0
    if opt.haze:
        d = f'{opt.beta},{opt.a}'
    else:
        d = 0
    if opt.rain:
        e = 1
    else:
        e = 0
    return f'{a}_{b}_{c}_{d}_{e}'
    


def degradation(image, depth, opt):
    if opt.rain:
        image = Rain(image)
    if opt.haze:
        image = Haze(image, depth, opt.beta, 0.05, opt.a, 0.1)
    if opt.downscale:
        image = Downscale(image, opt.scale)
    if opt.blur:
        image = Blur(image, opt.kernelsize, opt.angle)
    if opt.noise:
        image = Noise(image, opt.noisemode, opt.var)
    return image
    
    

if __name__ == "__main__":
    main()