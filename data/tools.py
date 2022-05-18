import os, argparse
from tkinter import E
import numpy as np
import glob
from LinearMotionBlur import LinearMotionBlur
import random
import albumentations as A
from PIL import Image


def Rain(image):
    transform = A.Compose(
    [A.RandomRain(always_apply=False, p=1.0, slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(183, 177, 162), blur_value=1, brightness_coefficient=1.0, rain_type=None)]
    )
    rainy = transform(image=image)
    image = rainy['image']
    return image

def Haze(image, depth, beta, bias, a, abias):
    beta_ = random.uniform(beta-bias, beta+bias)

    transmission_map = np.exp(-beta_ * depth)
    
    abias = 0.1
    a_ = random.uniform(a-abias, a+abias)
    A = [a_, a_, a_]

    h, w, _ = image.shape

    rep_atmosphere = np.tile(np.reshape(A, [1, 1, 3]), [h, w, 1])
    transmission_map = np.reshape(transmission_map, [h, w, 1])

    transmission_max = np.tile(transmission_map, [1, 1, 3])

    image = image * transmission_max + rep_atmosphere * (1 - transmission_max)
    return image


def Downscale(image, scale):
    h, w, _ = image.shape
    PILimage = Image.fromarray((image*255).astype('uint8'))
    PILimage = PILimage.resize((int(w/scale), int(h/scale)), Image.LANCZOS)
    image = np.array(PILimage).astype('float32')
    image = image/255
    return image

def Blur(image, kernel_size, angle):
    image = LinearMotionBlur(image, kernel_size, angle, 'full')
    image = np.array(image)
    return image

def Noise(image, mode, var):
    w, h, c = image.shape
    if mode == 'gaussian':
        mean = 0
        var = 0.025
        sigma = var ** 0.5

        gauss = np.random.normal(mean,sigma,(w,h,c)).astype('float32')
        image = image + gauss
    elif mode == 'poisson':
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        image = np.where(image<0, 0, image)
        image = np.random.poisson(image * vals) / float(vals)
    return image






