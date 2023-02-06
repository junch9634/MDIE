import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models, transforms
import numpy as np
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt
import argparse, os, math, random, time
from model.uien import *
from utils.loader import NYUv2
from utils.metrics import *
from utils.image_transform import *
from utils.loss import MSE_and_SSIM_loss

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--batchSize", type=int, default=8, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=200, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--gpus", default="1", type=str, help="gpu ids (default: 0)")
parser.add_argument("--basedir", default="os.path.join('result', 'each', 'single')", type=str, help="dataset base directory path")

# 0 deblur, 1 noise, 2 haze, 3 rain

def main():
    global opt, model
    opt = parser.parse_args()

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
    train_set = NYUv2(dir='/ailab_mat/dataset/NYU2v', mode='train', trans=transforms.ToTensor(), gt_crop_size=None)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, \
        batch_size=opt.batchSize, shuffle=True)
    test_set = NYUv2(dir='/ailab_mat/dataset/NYU2v', mode='val', trans=transforms.ToTensor(), gt_crop_size=None)
    test_data_loader = DataLoader(dataset=test_set, num_workers=1, \
        batch_size=opt.batchSize, shuffle=False)

    print("===> Building model")
    model = UIEN_SINGLE()
    criterion = MSE_and_SSIM_loss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    deblur_model = torch.load('result/each/deblur/checkpoint/model_epoch_150.pth')
    deblur_dict_name = ['conv_first1', 'residual_layer1', 'encoder1', 'decoder1', 'up2_1', 'up4_1', 'fea_ex1', 'cbam1', 'recon_layer1', 'up1']
    deblur_dict = {k: v for k, v in deblur_model['model'].items() if k.split('.')[0] in deblur_dict_name}
    for k, v in deblur_model['model'].items():
        if k.split('.')[0] in ['conv_first', 'residual_layer']:
            new_key = k.split('.')[0] + '1.' + '.'.join(k.split('.')[1:])
            deblur_dict[new_key] = v

    noise_model = torch.load('result/each/denoise/checkpoint/model_epoch_100.pth')
    noise_dict_name = ['conv_first2', 'residual_layer2', 'encoder2', 'decoder2', 'up2_2', 'up4_2', 'fea_ex2', 'cbam2', 'recon_layer2', 'up2']
    noise_dict = {k: v for k, v in noise_model['model'].items() if k.split('.')[0] in noise_dict_name}
    for k, v in noise_model['model'].items():
        if k.split('.')[0] in ['conv_first', 'residual_layer']:
            new_key = k.split('.')[0] + '2.' + '.'.join(k.split('.')[1:])
            noise_dict[new_key] = v

    haze_model = torch.load('result/each/dehaze/checkpoint/model_epoch_200.pth')
    haze_dict_name = ['conv_first3', 'residual_layer3', 'encoder3', 'decoder3', 'up2_3', 'up4_3', 'fea_ex3', 'cbam3', 'recon_layer3', 'up3']
    haze_dict = {k: v for k, v in haze_model['model'].items() if k.split('.')[0] in haze_dict_name}
    for k, v in haze_model['model'].items():
        if k.split('.')[0] in ['conv_first', 'residual_layer']:
            new_key = k.split('.')[0] + '3.' + '.'.join(k.split('.')[1:])
            haze_dict[new_key] = v

    rain_model = torch.load('result/each/derain/checkpoint/model_epoch_150.pth')
    rain_dict_name = ['conv_first4', 'residual_layer4', 'encoder4', 'decoder4', 'up2_4', 'up4_4', 'fea_ex4', 'cbam4', 'recon_layer4', 'up4']
    rain_dict = {k: v for k, v in rain_model['model'].items() if k.split('.')[0] in rain_dict_name}
    for k, v in rain_model['model'].items():
        if k.split('.')[0] in ['conv_first', 'residual_layer']:
            new_key = k.split('.')[0] + '4.' + '.'.join(k.split('.')[1:])
            rain_dict[new_key] = v

    model_state_dict = model.state_dict()
    model_state_dict.update(deblur_dict)
    model_state_dict.update(noise_dict)
    model_state_dict.update(haze_dict)
    model_state_dict.update(rain_dict)
    model.load_state_dict(model_state_dict)

    total_dict_name = deblur_dict_name + noise_dict_name + haze_dict_name + rain_dict_name
    for name , child in model.named_children():
        for param in child.parameters():
            if name.split('.')[0] in total_dict_name:
                param.requires_grad = False

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
            save_checkpoint(model, epoch, opt.basedir)
        train_avg_loss.append(train_loss)
        train_avg_psnr.append(train_psnr)
        train_avg_ssim.append(train_ssim)
        save_graph(train_avg_loss, train_avg_psnr, train_avg_ssim, epoch, 'train', opt.basedir)

        eval_loss, eval_psnr, eval_ssim = eval(test_data_loader, model, criterion)
        print(eval_loss)
        eval_avg_loss.append(eval_loss)
        eval_avg_psnr.append(eval_psnr)
        eval_avg_ssim.append(eval_ssim)
        save_graph(eval_avg_loss, eval_avg_psnr, eval_avg_ssim, epoch, 'eval', opt.basedir)
            

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr 

def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    avg_loss_result = 0.0
    avg_psnr_result = 0.0
    avg_ssim_result = 0.0
    for iteration, batch in enumerate(training_data_loader):
        # 0 = deblur, 1 = denoise, 2 = dehaze, 3 = derain
        input, target, img_name = batch[0].float(), batch[1].float(), batch[2]

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        output = model(input)

        loss = criterion(output, target)

        output = output.cpu().data.numpy().transpose(0, 2, 3, 1).astype(np.float32)
        target = target.cpu().data.numpy().transpose(0, 2, 3, 1).astype(np.float32)

        ssim_result = SSIM_RGB(output, target, shave_border=5)

        output_y = RGB2YCBRC(output)
        target_y = RGB2YCBRC(target)
        psnr_result = PSNR_Y(output_y, target_y, shave_border=5)

        avg_loss_result += loss.cpu().item()
        avg_psnr_result += psnr_result
        avg_ssim_result += ssim_result

        optimizer.zero_grad()
        loss.sum().backward()
        
        optimizer.step()
        print("===> Epoch[{}]({}/{}): Loss: {:.5} PSNR: {:.4} SSIM: {:.4}".format(epoch, iteration, len(training_data_loader), loss.item(), psnr_result, ssim_result))

        
    return avg_loss_result/len(training_data_loader), avg_psnr_result/len(training_data_loader), avg_ssim_result/len(training_data_loader)



def removeALLFile(filepath):
    if os.path.exists(filepath):
        for file in os.scandir(filepath):
            os.remove(file.path)
        return 'Remove All File'
    else:
        return 'Directory Not Found'

def eval(test_data_loader, model, criterion):
    os.makedirs('output', exist_ok=True)

    model.eval()
    avg_loss_predicted = 0.0
    avg_psnr_predicted = 0.0
    avg_ssim_predicted = 0.0
    avg_elapsed_time = 0.0

    samples_dir_name = os.path.join('result', 'each', 'all', 'samples')
    os.makedirs(samples_dir_name, exist_ok=True)
    print(removeALLFile(samples_dir_name))

    for iteration, batch in enumerate(test_data_loader): 
        input, target, img_name = batch[0].float(), batch[1].float(), batch[2]

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        start_time = time.time()
        with torch.no_grad():
            output = model(input)
        elapsed_time = time.time() - start_time
        avg_elapsed_time += elapsed_time

        loss = criterion(output, target)

        output = output.cpu().data.numpy().transpose(0, 2, 3, 1).astype(np.float32)
        target = target.cpu().data.numpy().transpose(0, 2, 3, 1).astype(np.float32)
        
        ssim_predicted = SSIM_RGB(output, target, shave_border=5)

        output_y = RGB2YCBRC(output)
        target_y = RGB2YCBRC(target)
        psnr_predicted = PSNR_Y(output_y, target_y, shave_border=5)

        avg_loss_predicted += loss.cpu().item()
        avg_psnr_predicted += psnr_predicted
        avg_ssim_predicted += ssim_predicted

        if iteration % 10 == 0:
            cv2.imwrite(os.path.join(samples_dir_name, img_name[0].split('/')[-1].split('.')[0] + '_' + str(psnr_predicted) + '.jpg'), output[0]*255)


    print("loss_predicted=", avg_loss_predicted/len(test_data_loader))
    print("PSNR_predicted=", avg_psnr_predicted/len(test_data_loader))
    print("SSIM_predicted=", avg_ssim_predicted/len(test_data_loader))
    print("It takes average {}s for processing".format(avg_elapsed_time/len(test_data_loader)))
    return avg_loss_predicted/len(test_data_loader), avg_psnr_predicted/len(test_data_loader), avg_ssim_predicted/len(test_data_loader)


def num2mode(num):
    mode_list = ['deblur', 'denoise', 'dehaze', 'derain']
    mode = mode_list[num]
    return mode

def save_checkpoint(model, epoch, base_dir):
    model_out_dir_path =os.path.join(base_dir, 'checkpoint')
    os.makedirs(model_out_dir_path, exist_ok=True)

    state = {"epoch": epoch ,"model": model.state_dict()}
    torch.save(state, os.path.join(model_out_dir_path, "model_epoch_{}.pth".format(epoch)))

    print("Checkpoint saved to {}".format(model_out_dir_path, "model_epoch_{}.pth".format(epoch)))

def save_graph(loss, psnr, ssim, epoch, mode, base_dir):
    graph_path_dir = os.path.join(base_dir, 'graph')
    os.makedirs(graph_path_dir, exist_ok=True)

    # loss graph
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Loss_{}'.format(epoch))
    if opt.resume:
        plt.plot(list(range(1, epoch - int(opt.resume.split('_')[-1].split('.')[0]) + 1)), loss)
    else:
        plt.plot(list(range(1, epoch + 1)), loss)

    plt.legend(['train'])
    plt.savefig(os.path.join(graph_path_dir, f'{mode}_loss.png'))
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
    plt.savefig(os.path.join(graph_path_dir, f'{mode}_psnr.png'))
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
    plt.savefig(os.path.join(graph_path_dir, f'{mode}_ssim.png'))
    plt.clf()

if __name__ == "__main__":
    main()