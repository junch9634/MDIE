import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.uien import UIEN_MULT, UIEN_SINGLE
# from dataset import DatasetFromHdf5
from utils.loader import REDS_Dataset_MULT
from torchvision import models, transforms
import torch.utils.model_zoo as model_zoo
# import matlab.engine
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.loss import MSE_and_SSIM_loss
from skimage.metrics import structural_similarity as ssim
import time

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--batchSize", type=int, default=2, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=200, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")
parser.add_argument("--gpus", default="2", type=str, help="gpu ids (default: 0)")
parser.add_argument("--neighbor", default="5", type=int, help="neighbor image ")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--mode", default=4, type=int)
parser.add_argument("--model_type", default='multi-tot-avg', type=str)

# single, multi-last-avg, multi-tot-avg, multi-tot-attn

# 0 deblur, 1 noise, 2 haze, 3 rain, 4 total

def main():
    global opt, model, netContent
    opt = parser.parse_args()

    opt.save_dir = os.path.join('result/%s/%s' %(num2mode(opt.mode), opt.model_type))
    os.makedirs(opt.save_dir, exist_ok=True)

    opt.cuda = True
    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    # print(opt.threads, opt.batchSize)
    train_set = REDS_Dataset_MULT(dir='/ailab_mat/dataset/NYU2v', mode='train', trans=transforms.ToTensor(), gt_crop_size=None)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, \
        batch_size=opt.batchSize, shuffle=True)
    test_set = REDS_Dataset_MULT(dir='/ailab_mat/dataset/NYU2v', mode='val', trans=transforms.ToTensor(), gt_crop_size=None)
    test_data_loader = DataLoader(dataset=test_set, num_workers=1, \
        batch_size=opt.batchSize, shuffle=False)

    print("===> Building model")
    
    if opt.model_type == 'single':
        model = UIEN_SINGLE()
    else:
        fusion_type = opt.model_type.split('-')[-1]
        model = UIEN_MULT(fusion_type=fusion_type)

    criterion = MSE_and_SSIM_loss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'])
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    train_avg_loss = []
    train_avg_psnr = []
    train_avg_ssim = []
    eval_avg_loss = []
    eval_avg_psnr = []
    eval_avg_ssim = []
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train_loss, train_psnr, train_ssim = train(training_data_loader, optimizer, model, criterion, epoch)
        if epoch % 10 == 0:
            save_checkpoint(model, epoch)

        train_avg_loss.append(train_loss)
        train_avg_psnr.append(train_psnr)
        train_avg_ssim.append(train_ssim)
        save_graph(train_avg_loss, train_avg_psnr, train_avg_ssim, epoch, 'train')

        eval_loss, eval_psnr, eval_ssim = eval(test_data_loader, model, criterion)
        eval_avg_loss.append(eval_loss)
        eval_avg_psnr.append(eval_psnr)
        eval_avg_ssim.append(eval_ssim)
        save_graph(eval_avg_loss, eval_avg_psnr, eval_avg_ssim, epoch, 'eval')
            

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr 

def PSNR(pred_, gt_, shave_border=5):
    psnrs = []
    for pred, gt in zip(pred_, gt_):
        pred = np.swapaxes(pred, 0, 2)*255
        gt = np.swapaxes(gt, 0, 2)*255

        height, width = pred.shape[:2]
        pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
        gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
        imdff = pred.astype(int) - gt.astype(int)
        rmse = math.sqrt(np.mean(imdff ** 2))
        if rmse == 0:
            return 100
        psnrs.append(20 * math.log10(255.0 / rmse))
    return sum(psnrs)/len(psnrs)


def SSIM(pred_, gt_, shave_border=5):
    ssims = []
    for pred, gt in zip(pred_, gt_):
        pred = np.moveaxis(np.squeeze(pred), 0, -1)
        gt = np.moveaxis(np.squeeze(gt), 0, -1)
        
        height, width = pred.shape[:2]
        pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
        gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]

        grayA = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        (score, diff) = ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        ssims.append(score)
    return sum(ssims)/len(ssims)


def PSNR_(pred_list, gt_list, shave_border=5):
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


def RGB2YCBRC(img_list, variable_type):
    # if variable_type == 'tensor':
    #     img = img.data[0].numpy().transpose(1,2,0).astype(np.float32)
    # elif variable_type == 'numpy':
    #     img = img

    out_list = []
    for img in img_list:
        img = img*255.
        img = np.clip(img, 0., 255.)

        im_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        im_ycbcr = np.array(im_ycbcr).astype(np.float32)#  * 255.
        im_y = im_ycbcr[:,:,0]

        out_list.append(im_y)
    out_list = np.array(out_list)
    return out_list



def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    losses = []
    psnrs = []
    ssims = []
    for iteration, batch in enumerate(training_data_loader):
        # 0 = deblur, 1 = denoise, 2 = dehaze, 3 = derain
        input, middle_GT, final_GT, img_name = batch
        input = input.cuda()

        middle_GT = [middle.cuda() for middle in middle_GT]
        final_GT = final_GT.cuda()

        # Forward
        optimizer.zero_grad()
        out, middle_output = model(input)

        # Loss
        loss_middle = 0.
        if (opt.model_type == 'single') or ('multi-last' in opt.model_type):
            alpha = 0.0
        else:
            alpha = 0.3
            for m_O, m_G in zip(middle_output, middle_GT):
                loss_middle += criterion(m_O, m_G) / len(middle_GT)
        loss_out = criterion(out, final_GT)
        loss = loss_middle * alpha + loss_out * (1-alpha)
        loss.backward()
        optimizer.step()
        
        # PSNR and SSIM
        psnr_result = PSNR(out.cpu().detach().numpy().astype(np.float32),
                           final_GT.cpu().detach().numpy().astype(np.float32),
                           shave_border=opt.scale)

        ssim_result = SSIM(out.cpu().detach().numpy().astype(np.float32),
                           final_GT.cpu().detach().numpy().astype(np.float32))

        losses.append(loss.item())
        psnrs.append(psnr_result)
        ssims.append(ssim_result)

        print("===> Epoch[{}]({}/{}): Loss: {:.5} PSNR: {:.4} SSIM: {:.4}".format(epoch, iteration, len(training_data_loader), loss.sum().item(), psnr_result, ssim_result))
        torch.cuda.empty_cache()

    return np.average(losses), np.average(psnrs), np.average(ssims)
    

def eval(test_data_loader, model, criterion):
    os.makedirs(os.path.join(opt.save_dir, 'samples'), exist_ok=True)

    model.eval()
    avg_loss_predicted = 0.0
    avg_psnr_predicted = 0.0
    avg_ssim_predicted = 0.0
    avg_elapsed_time = 0.0

    for _, batch in enumerate(test_data_loader): 
        input, middle_GT, final_GT, img_name = batch
        input = input.cuda()

        middle_GT = [middle.cuda() for middle in middle_GT]
        final_GT = final_GT.cuda()

        # Forward
        start_time = time.time()

        with torch.no_grad():
            out, _ = model(input)

        # Loss
        loss = criterion(out, final_GT)

        elapsed_time = time.time() - start_time
        avg_elapsed_time += elapsed_time

        avg_loss_predicted += loss.item()
        out = out.cpu().data.numpy().transpose(0, 2, 3, 1).astype(np.float32)
        final_GT = final_GT.cpu().data.numpy().transpose(0, 2, 3, 1).astype(np.float32)
        im_h_y = RGB2YCBRC(out, 'numpy')
        im_gt_y = RGB2YCBRC(final_GT, 'numpy')

        psnr_predicted = PSNR_(im_gt_y, im_h_y, shave_border=opt.scale)
        avg_psnr_predicted += psnr_predicted

        # ssim_predicted = SSIM(middle_output, final_GT)
        avg_ssim_predicted += 0.
        cv2.imwrite(os.path.join(opt.save_dir, 'samples', img_name[0].split('/')[-1].split('.')[0] + '_' + str(psnr_predicted) + '.jpg'), out[0]*255)
        
    print("loss_predicted=", avg_loss_predicted/len(test_data_loader))
    print("PSNR_predicted=", avg_psnr_predicted/len(test_data_loader))
    print("SSIM_predicted=", avg_ssim_predicted/len(test_data_loader))
    print("It takes average {}s for processing".format(avg_elapsed_time/len(test_data_loader)))
    return avg_loss_predicted/len(test_data_loader), avg_psnr_predicted/len(test_data_loader), avg_ssim_predicted/len(test_data_loader)


def num2mode(num):
    mode_list = ['deblur', 'denoise', 'dehaze', 'derain', 'total']
    mode = mode_list[num]
    return mode

def save_checkpoint(model, epoch):
    model_out_path = os.path.join(opt.save_dir, 'checkpoint', "model_epoch_{}.pth".format(epoch))
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)

    state = {"epoch": epoch ,"model": model.state_dict()}
    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

def save_graph(loss, psnr, ssim, epoch, mode):
    graph_folder = os.path.join(opt.save_dir, 'graph')
    os.makedirs(graph_folder, exist_ok=True)

    # loss graph
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Loss_{}'.format(epoch))
    if opt.resume:
        plt.plot(list(range(1, epoch - int(opt.resume.split('_')[-1].split('.')[0]) + 1)), loss)
    else:
        plt.plot(list(range(1, epoch + 1)), loss)

    plt.legend(['train'])
    plt.savefig(os.path.join(graph_folder, '{}_loss.png'.format(mode)))
    plt.clf()

    # psnr graph
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('PSNR_{}'.format(epoch))
    if opt.resume:
        plt.plot(list(range(1, epoch - int(opt.resume.split('_')[-1].split('.')[0]) + 1)), psnr)
    else:
        plt.plot(list(range(1, epoch + 1)), psnr)
    plt.legend(['train'])
    plt.savefig(os.path.join(graph_folder, '{}_psnr.png'.format(mode)))
    plt.clf()

    # ssim graph
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM_{}'.format(epoch))
    if opt.resume:
        plt.plot(list(range(1, epoch - int(opt.resume.split('_')[-1].split('.')[0]) + 1)), ssim)
    else:
        plt.plot(list(range(1, epoch + 1)), ssim)
    plt.legend(['train'])
    plt.savefig(os.path.join(graph_folder, '{}_ssim.png'.format(mode)))
    plt.clf()

if __name__ == "__main__":
    main()