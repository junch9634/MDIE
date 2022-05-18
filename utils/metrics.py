import numpy as np
import math
import cv2
from skimage.metrics import structural_similarity as ssim


def PSNR_RGB(pred_list, gt_list, shave_border=5):
    psnrs = []
    for pred, gt in zip(pred_list, gt_list):
        height, width = pred.shape[:2]
        pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
        gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
        imdff = pred.astype(int) - gt.astype(int)
        rmse = math.sqrt(np.mean(imdff ** 2))
        if rmse == 0:
            return 100
        psnrs.append(20 * math.log10(255.0 / rmse))
    return sum(psnrs)/len(psnrs)

def SSIM_RGB(pred_, gt_, shave_border=5):
    ssims = []
    for pred, gt in zip(pred_, gt_):
        height, width = pred.shape[:2]
        pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
        gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]

        grayA = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        (score, diff) = ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        ssims.append(score)
    return sum(ssims)/len(ssims)


def PSNR_Y(pred_list, gt_list, shave_border=5):
    out = 0.
    for pred, gt in zip(pred_list, gt_list):
        height, width = pred.shape
        pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
        gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
        imdff = pred - gt
        rmse = math.sqrt(np.mean(imdff ** 2))
        if rmse == 0:
            out += 100
        else:
            out += 20 * math.log10(255.0 / rmse)
    return out / pred_list.shape[0]
