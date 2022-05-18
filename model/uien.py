import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable, Function
import math
import numpy as np
# from modules import ConvOffset2d
import time
from .cbam import CBAM
from .unet import Encoder, Decoder


class Res_Block(nn.Module):
    def __init__(self):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res


def default_conv(in_channelss, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        modules = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                modules.append(conv(n_feat, 4 * n_feat, 3, bias))
                modules.append(nn.PixelShuffle(2))
                if bn: modules.append(nn.BatchNorm2d(n_feat))
                if act: modules.append(act())
        elif scale == 3:
            modules.append(conv(n_feat, 9 * n_feat, 3, bias))
            modules.append(nn.PixelShuffle(3))
            if bn: modules.append(nn.BatchNorm2d(n_feat))
            if act: modules.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*modules)



class UIEN(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=64, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3, reduction=4, bias=False):
        super(UIEN, self).__init__()
        self.name = 'TDAN'
        # Feature Extracter
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer = self.make_layer(Res_Block, 5)
        self.relu = nn.ReLU(inplace=True)
        self.conv_first1 = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer1 = self.make_layer(Res_Block, 5)
        self.conv_first2 = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer2 = self.make_layer(Res_Block, 5)
        self.conv_first3 = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer3 = self.make_layer(Res_Block, 5)
        self.conv_first4 = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer4 = self.make_layer(Res_Block, 5)

        # IEN                                                                                       n_feat = 40
        act=nn.PReLU()
        self.encoder1 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.encoder2 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.encoder3 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.encoder4 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder1 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.decoder2 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.decoder3 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.decoder4 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        
        up2 = [
            Upsampler(default_conv, 2, 84, act=False),
            nn.Conv2d(84, 64, 3, padding=1, bias=False)]
        self.up2_1 = nn.Sequential(*up2)
        self.up2_2 = nn.Sequential(*up2)
        self.up2_3 = nn.Sequential(*up2)
        self.up2_4 = nn.Sequential(*up2)

        up4 = [
            Upsampler(default_conv, 4, 104, act=False),
            nn.Conv2d(104, 64, 3, padding=1, bias=False)]
        self.up4_1 = nn.Sequential(*up4)
        self.up4_2 = nn.Sequential(*up4)
        self.up4_3 = nn.Sequential(*up4)
        self.up4_4 = nn.Sequential(*up4)


        fea_ex = [nn.Conv2d(64 * 3, 64, 3, padding= 1, bias=True),
                       nn.ReLU()]
        self.fea_ex1 = nn.Sequential(*fea_ex)
        self.fea_ex2 = nn.Sequential(*fea_ex)
        self.fea_ex3 = nn.Sequential(*fea_ex)
        self.fea_ex4 = nn.Sequential(*fea_ex)
        
        self.cbam1 = CBAM(64, 16)
        self.cbam2 = CBAM(64, 16)
        self.cbam3 = CBAM(64, 16)
        self.cbam4 = CBAM(64, 16)
        
        self.recon_layer1 = self.make_layer(Res_Block, 10)
        self.recon_layer2 = self.make_layer(Res_Block, 10)
        self.recon_layer3 = self.make_layer(Res_Block, 10)
        self.recon_layer4 = self.make_layer(Res_Block, 10)
          
        upscaling = [
            Upsampler(default_conv, 4, 64, act=False),
            nn.Conv2d(64, 3, 3, padding=1, bias=False)]

        self.up1 = nn.Sequential(*upscaling)
        self.up2 = nn.Sequential(*upscaling)
        self.up3 = nn.Sequential(*upscaling)
        self.up4 = nn.Sequential(*upscaling)

        self.recon_layer = self.make_layer(Res_Block, 10)
        self.up = nn.Sequential(*upscaling)

    #     # xavier initialization
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def module_1(self, x):
        x = self.encoder1(x)
        x = self.decoder1(x)
        
        out1 = x[0]
        out2 = self.up2_1(x[1])
        out3 = self.up4_1(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex1(x)
        return x

    def module_2(self, x):
        x = self.encoder2(x)
        x = self.decoder2(x)
        
        out1 = x[0]
        out2 = self.up2_2(x[1])
        out3 = self.up4_2(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex2(x)

        return x
    def module_3(self, x):
        x = self.encoder3(x)
        x = self.decoder3(x)
        
        out1 = x[0]
        out2 = self.up2_3(x[1])
        out3 = self.up4_3(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex3(x)
        return x

    def module_4(self, x):
        x = self.encoder4(x)
        x = self.decoder4(x)
        
        out1 = x[0]
        out2 = self.up2_4(x[1])
        out3 = self.up4_4(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex4(x)
        return x
        

    def forward(self, x):
        # b, 5, 3, 1280 720
        batch_size, ch, w, h = x.size()  # 5 video frames
        # center frame interpolation
        
        # extract features
        y = x.view(-1, ch, w, h)

        out = self.relu(self.conv_first(y))
        out = self.residual_layer(out)
        out = out.view(batch_size, -1, w, h)

        out1 = self.module_1(out)
        out1 = self.cbam1(out1)

        out2 = self.module_2(out)
        out2 = self.cbam2(out2)

        out3 = self.module_3(out)
        out3 = self.cbam3(out3)

        out4 = self.module_4(out)
        out4 = self.cbam4(out4)

        # out = torch.cat((out1, out2, out3, out4), dim=1)

        out = (out1 + out2 + out3 + out4) / 4

        out = self.recon_layer(out)
        out = self.up(out)
        
        return out



class UIEN_param(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=64, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3, reduction=4, bias=False):
        super(UIEN_param, self).__init__()
        self.name = 'TDAN'
        # Feature Extracter
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer = self.make_layer(Res_Block, 5)
        self.relu = nn.ReLU(inplace=True)
        self.conv_first1 = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer1 = self.make_layer(Res_Block, 5)
        self.conv_first2 = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer2 = self.make_layer(Res_Block, 5)
        self.conv_first3 = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer3 = self.make_layer(Res_Block, 5)
        self.conv_first4 = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer4 = self.make_layer(Res_Block, 5)

        # IEN                                                                                       n_feat = 40
        act=nn.PReLU()
        self.encoder1 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.encoder2 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.encoder3 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.encoder4 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder1 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.decoder2 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.decoder3 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.decoder4 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        
        up2 = [
            Upsampler(default_conv, 2, 84, act=False),
            nn.Conv2d(84, 64, 3, padding=1, bias=False)]
        self.up2_1 = nn.Sequential(*up2)
        self.up2_2 = nn.Sequential(*up2)
        self.up2_3 = nn.Sequential(*up2)
        self.up2_4 = nn.Sequential(*up2)

        up4 = [
            Upsampler(default_conv, 4, 104, act=False),
            nn.Conv2d(104, 64, 3, padding=1, bias=False)]
        self.up4_1 = nn.Sequential(*up4)
        self.up4_2 = nn.Sequential(*up4)
        self.up4_3 = nn.Sequential(*up4)
        self.up4_4 = nn.Sequential(*up4)


        fea_ex = [nn.Conv2d(64 * 3, 64, 3, padding= 1, bias=True),
                       nn.ReLU()]
        self.fea_ex1 = nn.Sequential(*fea_ex)
        self.fea_ex2 = nn.Sequential(*fea_ex)
        self.fea_ex3 = nn.Sequential(*fea_ex)
        self.fea_ex4 = nn.Sequential(*fea_ex)
        
        self.cbam1 = CBAM(64, 16)
        self.cbam2 = CBAM(64, 16)
        self.cbam3 = CBAM(64, 16)
        self.cbam4 = CBAM(64, 16)
        
        self.recon_layer1 = self.make_layer(Res_Block, 10)
        self.recon_layer2 = self.make_layer(Res_Block, 10)
        self.recon_layer3 = self.make_layer(Res_Block, 10)
        self.recon_layer4 = self.make_layer(Res_Block, 10)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 1)
        self.fc2 = nn.Linear(64, 1)
        self.fc3 = nn.Linear(64, 1)
        self.fc4 = nn.Linear(64, 1)
        self.sf = nn.Softmax(dim=1)

        upscaling = [
            Upsampler(default_conv, 4, 64, act=False),
            nn.Conv2d(64, 3, 3, padding=1, bias=False)]

        self.up1 = nn.Sequential(*upscaling)
        self.up2 = nn.Sequential(*upscaling)
        self.up3 = nn.Sequential(*upscaling)
        self.up4 = nn.Sequential(*upscaling)

        self.recon_layer = self.make_layer(Res_Block, 10)
        self.up = nn.Sequential(*upscaling)

    #     # xavier initialization
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def module_1(self, x):
        x = self.encoder1(x)
        x = self.decoder1(x)
        
        out1 = x[0]
        out2 = self.up2_1(x[1])
        out3 = self.up4_1(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex1(x)
        return x

    def module_2(self, x):
        x = self.encoder2(x)
        x = self.decoder2(x)
        
        out1 = x[0]
        out2 = self.up2_2(x[1])
        out3 = self.up4_2(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex2(x)

        return x
    def module_3(self, x):
        x = self.encoder3(x)
        x = self.decoder3(x)
        
        out1 = x[0]
        out2 = self.up2_3(x[1])
        out3 = self.up4_3(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex3(x)
        return x

    def module_4(self, x):
        x = self.encoder4(x)
        x = self.decoder4(x)
        
        out1 = x[0]
        out2 = self.up2_4(x[1])
        out3 = self.up4_4(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex4(x)
        return x
        

    def forward(self, x):
        # b, 5, 3, 1280 720
        batch_size, ch, w, h = x.size()  # 5 video frames
        # center frame interpolation
        
        # extract features
        y = x.view(-1, ch, w, h)

        out = self.relu(self.conv_first(y))
        out = self.residual_layer(out)
        out = out.view(batch_size, -1, w, h)

        out1 = self.module_1(out)
        out1 = self.cbam1(out1)

        out2 = self.module_2(out)
        out2 = self.cbam2(out2)

        out3 = self.module_3(out)
        out3 = self.cbam3(out3)

        out4 = self.module_4(out)
        out4 = self.cbam4(out4)

        a = self.gap(out1)
        a = torch.squeeze(a)
        a = self.fc1(a)


        b = self.gap(out2)
        b = torch.squeeze(b)
        b = self.fc2(b)

        c = self.gap(out3)
        c = torch.squeeze(c)
        c = self.fc3(c)

        d = self.gap(out4)
        d = torch.squeeze(d)
        d = self.fc4(d)

        params = torch.cat((a, b, c, d), dim = 1)
        params = self.sf(params).T

        out1 = out1*params[0].view(-1, 1, 1, 1)
        out2 = out1*params[1].view(-1, 1, 1, 1)
        out3 = out1*params[2].view(-1, 1, 1, 1)
        out4 = out1*params[3].view(-1, 1, 1, 1)

        out = (out1 + out2 + out3 + out4)

        out = self.recon_layer(out)
        out = self.up(out)
        
        return out




class UIEN_sd(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=64, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3, reduction=4, bias=False):
        super(UIEN_sd, self).__init__()
        self.name = 'TDAN'
        # Feature Extracter
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer = self.make_layer(Res_Block, 5)
        self.relu = nn.ReLU(inplace=True)
        # IEN                                                                                       n_feat = 40
        act=nn.PReLU()
        self.encoder1 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.encoder2 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.encoder3 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.encoder4 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder1 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.decoder2 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.decoder3 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.decoder4 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        
        up2 = [
            Upsampler(default_conv, 2, 84, act=False),
            nn.Conv2d(84, 64, 3, padding=1, bias=False)]
        self.up2_1 = nn.Sequential(*up2)
        self.up2_2 = nn.Sequential(*up2)
        self.up2_3 = nn.Sequential(*up2)
        self.up2_4 = nn.Sequential(*up2)

        up4 = [
            Upsampler(default_conv, 4, 104, act=False),
            nn.Conv2d(104, 64, 3, padding=1, bias=False)]
        self.up4_1 = nn.Sequential(*up4)
        self.up4_2 = nn.Sequential(*up4)
        self.up4_3 = nn.Sequential(*up4)
        self.up4_4 = nn.Sequential(*up4)


        fea_ex = [nn.Conv2d(64 * 3, 64, 3, padding= 1, bias=True),
                       nn.ReLU()]
        self.fea_ex1 = nn.Sequential(*fea_ex)
        self.fea_ex2 = nn.Sequential(*fea_ex)
        self.fea_ex3 = nn.Sequential(*fea_ex)
        self.fea_ex4 = nn.Sequential(*fea_ex)
        
        self.cbam1 = CBAM(64, 16)
        self.cbam2 = CBAM(64, 16)
        self.cbam3 = CBAM(64, 16)
        self.cbam4 = CBAM(64, 16)
        
        self.recon_layer1 = self.make_layer(Res_Block, 10)
        self.recon_layer2 = self.make_layer(Res_Block, 10)
        self.recon_layer3 = self.make_layer(Res_Block, 10)
        self.recon_layer4 = self.make_layer(Res_Block, 10)
          
        upscaling = [
            Upsampler(default_conv, 4, 64, act=False),
            nn.Conv2d(64, 3, 3, padding=1, bias=False)]

        self.up1 = nn.Sequential(*upscaling)
        self.up2 = nn.Sequential(*upscaling)
        self.up3 = nn.Sequential(*upscaling)
        self.up4 = nn.Sequential(*upscaling)

    #     # xavier initialization
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def module_1(self, x):
        x = self.encoder1(x)
        x = self.decoder1(x)
        
        out1 = x[0]
        out2 = self.up2_1(x[1])
        out3 = self.up4_1(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex1(x)
        return x

    def module_2(self, x):
        x = self.encoder2(x)
        x = self.decoder2(x)
        
        out1 = x[0]
        out2 = self.up2_2(x[1])
        out3 = self.up4_2(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex2(x)

        return x
    def module_3(self, x):
        x = self.encoder3(x)
        x = self.decoder3(x)
        
        out1 = x[0]
        out2 = self.up2_3(x[1])
        out3 = self.up4_3(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex3(x)
        return x

    def module_4(self, x):
        x = self.encoder4(x)
        x = self.decoder4(x)
        
        out1 = x[0]
        out2 = self.up2_4(x[1])
        out3 = self.up4_4(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex4(x)
        return x
        

    def forward(self, x, mode):
        # b, 5, 3, 1280 720
        batch_size, ch, w, h = x.size()  # 5 video frames
        # center frame interpolation
        
        # extract features
        y = x.view(-1, ch, w, h)

        out = self.relu(self.conv_first(y))
        out = self.residual_layer(out)
        out = out.view(batch_size, -1, w, h)

        if mode == 0:
            out = self.module_1(out)
            out = self.cbam1(out)
            out = self.recon_layer1(out)
            out = self.up1(out)

        elif mode == 1:
            out = self.module_2(out)
            out = self.cbam2(out)
            out = self.recon_layer2(out)
            out = self.up2(out)

        elif mode == 2:    
            out = self.module_3(out)
            out = self.cbam3(out)
            out = self.recon_layer3(out)
            out = self.up3(out)

        elif mode == 3:    
            out = self.module_4(out)
            out = self.cbam4(out)
            out = self.recon_layer4(out)
            out = self.up4(out)
        
        else:
            raise('Select Proper Mode')
        
        # out = self.recon_layer(out)
        # out = self.up(out)
        
        return 0, out


class UIEN_double(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=64, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3, reduction=4, bias=False):
        super(UIEN_double, self).__init__()
        self.name = 'TDAN'
        # Feature Extracter
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer = self.make_layer(Res_Block, 5)
        self.relu = nn.ReLU(inplace=True)
        self.conv_first1 = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer1 = self.make_layer(Res_Block, 5)
        self.conv_first2 = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer2 = self.make_layer(Res_Block, 5)
        self.conv_first3 = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer3 = self.make_layer(Res_Block, 5)
        self.conv_first4 = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer4 = self.make_layer(Res_Block, 5)

        # IEN                                                                                       n_feat = 40
        act=nn.PReLU()
        self.encoder1 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.encoder2 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.encoder3 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.encoder4 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder1 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.decoder2 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.decoder3 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.decoder4 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        
        up2 = [
            Upsampler(default_conv, 2, 84, act=False),
            nn.Conv2d(84, 64, 3, padding=1, bias=False)]
        self.up2_1 = nn.Sequential(*up2)
        self.up2_2 = nn.Sequential(*up2)
        self.up2_3 = nn.Sequential(*up2)
        self.up2_4 = nn.Sequential(*up2)

        up4 = [
            Upsampler(default_conv, 4, 104, act=False),
            nn.Conv2d(104, 64, 3, padding=1, bias=False)]
        self.up4_1 = nn.Sequential(*up4)
        self.up4_2 = nn.Sequential(*up4)
        self.up4_3 = nn.Sequential(*up4)
        self.up4_4 = nn.Sequential(*up4)


        fea_ex = [nn.Conv2d(64 * 3, 64, 3, padding= 1, bias=True),
                       nn.ReLU()]
        self.fea_ex1 = nn.Sequential(*fea_ex)
        self.fea_ex2 = nn.Sequential(*fea_ex)
        self.fea_ex3 = nn.Sequential(*fea_ex)
        self.fea_ex4 = nn.Sequential(*fea_ex)
        
        self.cbam1 = CBAM(64, 16)
        self.cbam2 = CBAM(64, 16)
        self.cbam3 = CBAM(64, 16)
        self.cbam4 = CBAM(64, 16)
        
        self.recon_layer1 = self.make_layer(Res_Block, 10)
        self.recon_layer2 = self.make_layer(Res_Block, 10)
        self.recon_layer3 = self.make_layer(Res_Block, 10)
        self.recon_layer4 = self.make_layer(Res_Block, 10)
          
        upscaling = [
            Upsampler(default_conv, 4, 64, act=False),
            nn.Conv2d(64, 3, 3, padding=1, bias=False)]

        self.up1 = nn.Sequential(*upscaling)
        self.up2 = nn.Sequential(*upscaling)
        self.up3 = nn.Sequential(*upscaling)
        self.up4 = nn.Sequential(*upscaling)

        self.recon_layer = self.make_layer(Res_Block, 10)
        self.up = nn.Sequential(*upscaling)

    #     # xavier initialization
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def module_1(self, x):
        x = self.encoder1(x)
        x = self.decoder1(x)
        
        out1 = x[0]
        out2 = self.up2_1(x[1])
        out3 = self.up4_1(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex1(x)
        return x

    def module_2(self, x):
        x = self.encoder2(x)
        x = self.decoder2(x)
        
        out1 = x[0]
        out2 = self.up2_2(x[1])
        out3 = self.up4_2(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex2(x)

        return x
    def module_3(self, x):
        x = self.encoder3(x)
        x = self.decoder3(x)
        
        out1 = x[0]
        out2 = self.up2_3(x[1])
        out3 = self.up4_3(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex3(x)
        return x

    def module_4(self, x):
        x = self.encoder4(x)
        x = self.decoder4(x)
        
        out1 = x[0]
        out2 = self.up2_4(x[1])
        out3 = self.up4_4(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex4(x)
        return x
        

    def forward(self, x):
        # b, 5, 3, 1280 720
        batch_size, ch, w, h = x.size()  # 5 video frames
        # center frame interpolation
        
        # extract features
        y = x.view(-1, ch, w, h)

        out = self.relu(self.conv_first(y))
        out = self.residual_layer(out)
        out = out.view(batch_size, -1, w, h)

        out1 = self.module_1(out)
        out1 = self.cbam1(out1)

        out2 = self.module_2(out)
        out2 = self.cbam2(out2)

        out3 = self.module_3(out)
        out3 = self.cbam3(out3)

        out4 = self.module_4(out)
        out4 = self.cbam4(out4)

        # out = torch.cat((out1, out2, out3, out4), dim=1)

        out = (out1 + out2 + out3 + out4) / 4

        out = self.recon_layer(out)
        out = self.up(out)
        
        return out



class UIEN_MULT(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=64, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3, reduction=4, bias=False, fusion_type='avg'):
        super(UIEN_MULT, self).__init__()
        self.name = 'TDAN'
        # Feature Extracter
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer = self.make_layer(Res_Block, 5)
        self.relu = nn.ReLU(inplace=True)

        self.fusion_type = fusion_type

        # IEN                                                                     
        act=nn.PReLU()
        self.encoder1 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.encoder2 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.encoder3 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.encoder4 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder1 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.decoder2 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.decoder3 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.decoder4 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        
        up2 = [
            Upsampler(default_conv, 2, 84, act=False),
            nn.Conv2d(84, 64, 3, padding=1, bias=False)]
        self.up2_1 = nn.Sequential(*up2)
        self.up2_2 = nn.Sequential(*up2)
        self.up2_3 = nn.Sequential(*up2)
        self.up2_4 = nn.Sequential(*up2)

        up4 = [
            Upsampler(default_conv, 4, 104, act=False),
            nn.Conv2d(104, 64, 3, padding=1, bias=False)]
        self.up4_1 = nn.Sequential(*up4)
        self.up4_2 = nn.Sequential(*up4)
        self.up4_3 = nn.Sequential(*up4)
        self.up4_4 = nn.Sequential(*up4)


        fea_ex = [nn.Conv2d(64 * 3, 64, 3, padding= 1, bias=True),
                       nn.ReLU()]
        self.fea_ex1 = nn.Sequential(*fea_ex)
        self.fea_ex2 = nn.Sequential(*fea_ex)
        self.fea_ex3 = nn.Sequential(*fea_ex)
        self.fea_ex4 = nn.Sequential(*fea_ex)
        
        self.cbam1 = CBAM(64, 16)
        self.cbam2 = CBAM(64, 16)
        self.cbam3 = CBAM(64, 16)
        self.cbam4 = CBAM(64, 16)
        
        self.recon_layer1 = self.make_layer(Res_Block, 10)
        self.recon_layer2 = self.make_layer(Res_Block, 10)
        self.recon_layer3 = self.make_layer(Res_Block, 10)
        self.recon_layer4 = self.make_layer(Res_Block, 10)
        self.recon_layer_last = self.make_layer(Res_Block, 10)
          
        upscaling = [
            Upsampler(default_conv, 4, 64, act=False),
            nn.Conv2d(64, 3, 3, padding=1, bias=False)]

        self.up1 = nn.Sequential(*upscaling)
        self.up2 = nn.Sequential(*upscaling)
        self.up3 = nn.Sequential(*upscaling)
        self.up4 = nn.Sequential(*upscaling)
        self.up_last = nn.Sequential(*upscaling)

    #     # xavier initialization
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def module_1(self, x):
        x = self.encoder1(x)
        x = self.decoder1(x)
        
        out1 = x[0]
        out2 = self.up2_1(x[1])
        out3 = self.up4_1(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex1(x)
        return x

    def module_2(self, x):
        x = self.encoder2(x)
        x = self.decoder2(x)
        
        out1 = x[0]
        out2 = self.up2_2(x[1])
        out3 = self.up4_2(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex2(x)

        return x
    def module_3(self, x):
        x = self.encoder3(x)
        x = self.decoder3(x)
        
        out1 = x[0]
        out2 = self.up2_3(x[1])
        out3 = self.up4_3(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex3(x)
        return x

    def module_4(self, x):
        x = self.encoder4(x)
        x = self.decoder4(x)
        
        out1 = x[0]
        out2 = self.up2_4(x[1])
        out3 = self.up4_4(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex4(x)
        return x
        

    def forward(self, x):
        # b, 5, 3, 1280 720
        batch_size, ch, w, h = x.size()  # 5 video frames
        # center frame interpolation
        
        # extract features
        y = x.view(-1, ch, w, h)

        out = self.relu(self.conv_first(y))
        out = self.residual_layer(out)
        out = out.view(batch_size, -1, w, h)

        # Module 1
        out1 = self.module_1(out)
        out1 = self.cbam1(out1)
        out1_recon = self.recon_layer1(out1)
        out1_recon = self.up1(out1_recon)

        # Module 2
        out2 = self.module_2(out)
        out2 = self.cbam2(out2)
        out2_recon = self.recon_layer2(out2)
        out2_recon = self.up2(out2_recon)

        # Module 3
        out3 = self.module_3(out)
        out3 = self.cbam3(out3)
        out3_recon = self.recon_layer3(out3)
        out3_recon = self.up3(out3_recon)

        # Module 4
        out4 = self.module_4(out)
        out4 = self.cbam4(out4)
        out4_recon = self.recon_layer4(out4)
        out4_recon = self.up4(out4_recon)

        # Merge
        if self.fusion_type == 'avg':
            out = (out1 + out2 + out3 + out4) / 4
        elif self.fusion_type == 'attn':
            out = ''
        else:
            raise('Select Proper Fusion Type')

        out = self.recon_layer_last(out)
        out = self.up_last(out)
        return out, [out1_recon, out2_recon, out3_recon, out4_recon]


class UIEN_SINGLE(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=64, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3, reduction=4, bias=False):
        super(UIEN_SINGLE, self).__init__()
        self.name = 'TDAN'
        # Feature Extracter
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.residual_layer = self.make_layer(Res_Block, 5)
        self.relu = nn.ReLU(inplace=True)

        # IEN                                                                     
        act=nn.PReLU()
        self.encoder1 = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.decoder1 = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        
        up2 = [
            Upsampler(default_conv, 2, 84, act=False),
            nn.Conv2d(84, 64, 3, padding=1, bias=False)]
        self.up2_1 = nn.Sequential(*up2)

        up4 = [
            Upsampler(default_conv, 4, 104, act=False),
            nn.Conv2d(104, 64, 3, padding=1, bias=False)]
        self.up4_1 = nn.Sequential(*up4)


        fea_ex = [nn.Conv2d(64 * 3, 64, 3, padding= 1, bias=True),
                       nn.ReLU()]
        self.fea_ex1 = nn.Sequential(*fea_ex)
        self.cbam = CBAM(64, 16)
          
        upscaling = [
            Upsampler(default_conv, 4, 64, act=False),
            nn.Conv2d(64, 3, 3, padding=1, bias=False)]

        self.recon_layer_last = self.make_layer(Res_Block, 10)
        self.up_last = nn.Sequential(*upscaling)


    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
    
    def module_1(self, x):
        x = self.encoder1(x)
        x = self.decoder1(x)
        
        out1 = x[0]
        out2 = self.up2_1(x[1])
        out3 = self.up4_1(x[2])

        x = torch.cat([out1, out2, out3], dim=1)

        # reconstruction
        x = self.fea_ex1(x)
        return x
       

    def forward(self, x):
        # b, 5, 3, 1280 720
        batch_size, ch, w, h = x.size()  # 5 video frames
        # center frame interpolation
        
        # extract features
        y = x.view(-1, ch, w, h)

        out = self.relu(self.conv_first(y))
        out = self.residual_layer(out)
        out = out.view(batch_size, -1, w, h)

        # Module 1
        out = self.module_1(out)
        out = self.cbam(out)
        out = self.recon_layer_last(out)
        out = self.up_last(out)
        return out