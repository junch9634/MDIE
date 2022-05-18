# import matlab.engine
import argparse, os
from configparser import Interpolation
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import cv2
from torch.utils.data import DataLoader
from torchvision import models, transforms
from loader import REDS_Dataset

parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_srresnet.pth", type=str, help="model path")
# parser.add_argument("--dataset", default="Set5", type=str, help="dataset name, Default: Set5")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

def PSNR(pred, gt, shave_border=5):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def RGB2YCBRC(img, variable_type):
    # if variable_type == 'tensor':
    #     img = img.data[0].numpy().transpose(1,2,0).astype(np.float32)
    # elif variable_type == 'numpy':
    #     img = img

    img = img*255.
    img = np.clip(img, 0., 255.)

    im_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    im_ycbcr = np.array(im_ycbcr).astype(np.float32)#  * 255.
    im_y = im_ycbcr[:,:,0]
    return im_y
    

opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model = torch.load(opt.model)["model"]

test_set = REDS_Dataset(dir='/data/chang/data', mode='val', neighbor=1, trans=transforms.ToTensor(), gt_crop_size=None)
test_data_loader = DataLoader(dataset=test_set, num_workers=1, \
        batch_size=1, shuffle=False)

avg_psnr_predicted = 0.0
avg_psnr_bicubic = 0.0
avg_elapsed_time = 0.0

for iteration, batch in enumerate(test_data_loader): 
    im_l, im_gt, img_name = batch[0].float(), np.squeeze(batch[1].float().numpy(), axis=0).transpose(1,2,0), batch[2]
    h, w, c = im_gt.shape

    if cuda:
        model = model.cuda()
        im_input = im_l.cuda()
    else:
        im_input = im_l
        model = model.cpu()

    start_time = time.time()
    HR_4x = model(im_input)
    elapsed_time = time.time() - start_time
    avg_elapsed_time += elapsed_time

    HR_4x = HR_4x.cpu().data[0].numpy().transpose(1,2,0).astype(np.float32)
    im_h_y = RGB2YCBRC(HR_4x, 'tensor')
    im_gt_y = RGB2YCBRC(im_gt, 'numpy')

    psnr_predicted = PSNR(im_gt_y, im_h_y,shave_border=opt.scale)
    avg_psnr_predicted += psnr_predicted
    
    im_b_y = cv2.resize(np.squeeze(im_l.float().numpy()).transpose(1,2,0), dsize=(w,h), interpolation=cv2.INTER_CUBIC)
    psnr_bicubic = PSNR(im_gt, im_b_y,shave_border=opt.scale)
    avg_psnr_bicubic += psnr_bicubic
    
    cv2.imwrite(os.path.join('output', img_name[0].split('/')[-1].split('.')[0] + '_' + str(psnr_predicted) + '.jpg'), HR_4x*255)


# print("Scale=", opt.scale)
# print("Dataset=", opt.dataset)
print("PSNR_predicted=", avg_psnr_predicted/len(test_data_loader))
print("PSNR_bicubic=", avg_psnr_bicubic/len(test_data_loader))
print("It takes average {}s for processing".format(avg_elapsed_time/len(test_data_loader)))
