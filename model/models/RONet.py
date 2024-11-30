 # coding=utf-8
#from inspect import classify_class_attrs
#from select import select
#from turtle import forward
#from typing_extensions import Self
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.conv import Conv2d
#from model.resattention import res_cbam
import torchvision.models as models
#from model.res2fg import res2net
import torch.nn.functional as F
import math
from models.swinNet import SwinTransformer
import time
import math
from functools import partial
from typing import Optional, Callable
from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat






class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)*x

class ChannelAttention_1(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_1, self).__init__()
        
        #self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)#dim=1是因为传入数据是多加了一维unsqueeze(0)
        x=max_out
        x = self.conv1(x)
        return self.sigmoid(x)


#mid level
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class PFM(nn.Module):# Pyramid Feature Module
    def __init__(self,in_channels):
        super(PFM,self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(in_channels, in_channels, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=7, dilation=7)
        )
        self.conv = nn.Conv2d(4*in_channels,in_channels,kernel_size=3,padding=1)
    def forward(self,x):
        #x0,x1,x2,x3 = torch.split(x,x.size()[1]//4,dim=1)
        y0 = self.branch0(x)
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)

        y = self.conv(torch.cat((y0,y1,y2,y3),1))
        return y

class SplitAttention(nn.Module):
    def __init__(self,in_channel):
        super(SplitAttention,self).__init__()
        self.ca1 = ChannelAttention(in_channel)
        self.ca2 = ChannelAttention(in_channel)
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
    def forward(self,x):
        x0,x1,x2,x3 = torch.split(x,x.size()[1]//4,dim=1)
        v0 = self.ca1(x0)
        z0 = x0*v0+x0
        
        v2 = self.ca2(x2)
        z2 = x2*v2+x2

        v1 = self.sa1(x1)
        z1 = x1*v1+x1

        v3 = self.sa2(x3)
        z3 = x3*v3+x3

        z = torch.cat((z0,z1,z2,z3),1)
        return z

class CBR(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CBR,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.Ba = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.relu(self.Ba(self.conv(x)))

        return out
class RTI(nn.Module):
    def __init__(self,in_channels):
        super(RTI,self).__init__()
        self.ca_r = ChannelAttention_1(in_channels)
        self.ca_t = ChannelAttention_1(in_channels)
        self.sa_r = SpatialAttention()
        self.sa_t = SpatialAttention()
        self.conv = CBR(2*in_channels,in_channels)

        
    def forward(self,r,t,s):
        r = s*r+r
        t = t*s+t
        r_ca = r*self.ca_r(r)+r
        r_s = self.sa_r(r_ca)
        t_ca = t*self.ca_t(t)+t
        t_sa = t_ca*r_s+t_ca
        out = self.conv(torch.cat((t_sa,r_ca),1))

        

        return out
class SO(nn.Module):
    def __init__(self,in_channels):
        super(SO,self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1,groups=in_channels)
        self.conv0_1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1,groups=in_channels)
        self.conv1_0 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1,groups=in_channels)
        self.conv1_1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1,groups=in_channels)
        self.spatial0 = SpatialAttention()
        self.spatial1 = SpatialAttention()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self,x0,x1):
        y0 = self.conv0_1(self.conv0_0(x0))
        y1 = self.conv1_1(self.conv1_0(x1))
        out0 = self.spatial0(y0)*y0+y0
        out1 = self.spatial1(y1)*y1+y1
        if out0.size()== out1.size():
            out = out0+out1
        else:
            out = out0+self.upsample2(out1)
            
        return out
class RE(nn.Module):
    def __init__(self):
        super(RE,self).__init__()
        # self.conv_1x1_0 = nn.Conv2d(64,64,kernel_size=1)
        # self.conv_1x1_1 = nn.Conv2d(64,64,kernel_size=1)
        # self.conv_1x1_4 = nn.Conv2d(512,64,kernel_size=1)
        self.conv0 = nn.Conv2d(128,32,kernel_size=1)
        self.conv1 = nn.Conv2d(128,32,kernel_size=1)
        self.conv2 = nn.Conv2d(256,32,kernel_size=1)
        self.conv3 = nn.Conv2d(512,32,kernel_size=1)
        self.conv4 = nn.Conv2d(1024,32,kernel_size=1)
        self.RP0 = RP_low(32)
        self.RP1 = RP_mid(32)
        self.RP2 = RP_mid(32)
        self.RP3 = RP_high(32)
        self.RP4 = RP_high(32)

        self.SO0 = SO(32)
        self.SO1 = SO(32)
        self.SO2 = SO(32)
        self.SO3 = SO(32)

        self.conv1x1_0 = nn.Conv2d(32,1,kernel_size=1)
        self.conv1x1_1 = nn.Conv2d(32,1,kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(32,1,kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(32,1,kernel_size=1)




        
    def forward(self,x0,x1,x2,x3,x4):
        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        rp0 = self.RP0(x0)
        rp1 = self.RP1(x1)
        rp2 = self.RP2(x2)
        rp3 = self.RP3(x3)
        rp4 = self.RP4(x4)

        f3 = self.SO3(rp3,rp4)
        f2 = self.SO2(rp2,f3)
        f1 = self.SO1(rp1,f2)
        f0 = self.SO0(rp0,f1)

        s0 = self.conv1x1_1(f0)
        s1 = self.conv1x1_1(f1)
        s2 = self.conv1x1_2(f2)
        s3 = self.conv1x1_3(f3)


        
        
        return s0,s1,s2,s3




class PFM(nn.Module):# Pyramid Feature Module
    def __init__(self,in_channels):
        super(PFM,self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=2, dilation=2)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=4, dilation=4)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 1),
            BasicConv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(in_channels, in_channels, 3, padding=6, dilation=6)
        )
        self.sa0 = SpatialAttention()
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()

        self.conv = nn.Conv2d(4*in_channels,in_channels,kernel_size=3,padding=1,groups=in_channels)
        

    def forward(self,x):
        #x0,x1,x2,x3 = torch.split(x,x.size()[1]//4,dim=1)
        y0 = self.branch0(x)

        y1 = self.branch1(x+self.sa0(y0)*x)
        y2 = self.branch2(x+self.sa1(y1)*x)
        y3 = self.branch3(x+self.sa2(y2)*x)


        #y3 = self.branch0(x3*self.ca2(y2)+x3)

        
        out = self.conv(torch.cat((y0,y1,y2,y3),1))+x
        #y = self.conv(y)
        return out


class SI(nn.Module):#输入三通道
    def __init__(self, in_channels):
        super(SI, self).__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.down1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1,groups=in_channels)



        self.up2 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.down2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1,groups=in_channels)

        #MLP
        self.mlp_conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1,groups=in_channels)
        self.mlp_relu = nn.ReLU()
        self.mlp_conv2 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1,groups=in_channels)

        self.ca = ChannelAttention_1(in_channels)
        self.sa = SpatialAttention()
        self.conv3 = nn.Conv2d(3*in_channels,in_channels,kernel_size=3,padding=1,groups=in_channels)

    def forward(self, x):
        l = self.down1(self.conv1(self.up1(x)))
        h = self.up2(self.conv2(self.down2(x)))
        m = self.mlp_conv2(self.mlp_relu(self.mlp_conv1(x)))
        y = self.conv3(torch.cat((l,m,h),1))
        y = y+x
        y_ca = self.ca(y)*y+y
        out = self.sa(y_ca)*y_ca+y_ca
        return out

class RecurrrentAttention(nn.Module):
    def __init__(self,in_channels):
        super(RecurrrentAttention,self).__init__()

        self.ca_r1 = ChannelAttention(in_channels)
        self.sa_r1 = SpatialAttention()
        self.ca_t1 = ChannelAttention(in_channels)
        self.sa_t1 = SpatialAttention()
        self.sa_l = SpatialAttention()
        self.conv1 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1,groups=in_channels)
        
    def forward(self,r,t):
        r1 = self.ca_r1(r)*r+r
        t_r1 = self.sa_r1(r1)*t+t
        t1 = self.ca_t1(t_r1)*t_r1+t_r1
        r_t1 = self.sa_t1(t1)*r1+r1
        out = self.conv1(torch.cat((r_t1,t_r1),1))
        return out
class RONet(nn.Module):#输入三通道
    def __init__(self, in_channels):
        super(RONet, self).__init__()
        #resnet_RGB = models.resnet34(pretrained=True)
        #resnet_T = models.resnet34(pretrained=True)
        self.rgb_swin= SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        #self.depth_swin = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        #resnet2 = models.resnet18(pretrained=True)
        #self.weight=nn.Parameter(torch.FloatTensor(1))
        #
        #reanet = res2net()
        #res2n
        # ************************* Encoder ***************************
        # input conv3*3,64
        #self.conv_RGB = resnet_RGB.conv1
        #self.bn_RGB = resnet_RGB.bn1
        #self.relu_RGB = resnet_RGB.relu
        #self.maxpool_RGB = resnet_RGB.maxpool#计算得到112.5 但取112 向下取整
        # Extract Features
        #self.encoder1_RGB = resnet_RGB.layer1
        #self.encoder2_RGB = resnet_RGB.layer2
        #self.encoder3_RGB = resnet_RGB.layer3
        #self.encoder4_RGB = resnet_RGB.layer4
        
        #self.conv_T = resnet_T.conv1
        #self.bn_T = resnet_T.bn1
        #self.relu_T = resnet_T.relu
        #self.maxpool_T = resnet_T.maxpool#计算得到112.5 但取112 向下取整
        # Extract Features
        #self.encoder1_T = resnet_T.layer1
        #self.encoder2_T = resnet_T.layer2
        #self.encoder3_T = resnet_T.layer3
        #self.encoder4_T = resnet_T.layer4
        # ************************* Decoder ***************************
        self.re_RGB = RE()
        self.re_T = RE()
        self.re_RT = RE()

        

        self.conv_3x3_0_RGB = nn.Conv2d(128,32,1)
        self.conv_3x3_1_RGB = nn.Conv2d(128,32,1)
        self.conv_3x3_2_RGB = nn.Conv2d(256,32,1)
        self.conv_3x3_3_RGB = nn.Conv2d(512,32,1)
        self.conv_3x3_4_RGB = nn.Conv2d(1024,32,1)

        self.conv_3x3_0_T = nn.Conv2d(128,32,1)
        self.conv_3x3_1_T = nn.Conv2d(128,32,1)
        self.conv_3x3_2_T = nn.Conv2d(256,32,1)
        self.conv_3x3_3_T = nn.Conv2d(512,32,1)
        self.conv_3x3_4_T = nn.Conv2d(1024,32,1)

        self.conv_3x3_0_de = nn.Conv2d(32*2,32,1)
        self.conv_3x3_1_de= nn.Conv2d(32*2,32,1)
        self.conv_3x3_2_de = nn.Conv2d(32*2,32,1)
        self.conv_3x3_3_de = nn.Conv2d(32*2,32,1)

        #self.decoder_RGB = Decoder()
        #self.decoder_T = Decoder()
        self.ra0 = RecurrrentAttention(32)
        self.ra1 = RecurrrentAttention(32)
        self.ra2 = RecurrrentAttention(32)
        self.ra3 = RecurrrentAttention(32)
        self.ra4 = RecurrrentAttention(32)

        self.se3 = SI(32)
        self.se4 = SI(32)

        self.pfm0 = PFM(32)
        self.pfm1 = PFM(32)
        self.pfm2 = PFM(32)

        
        
        # ************************* Feature Map Upsample ***************************
        self.downsample1 = nn.Upsample(scale_factor=0.5, mode='bilinear')
        self.downsample2 = nn.Upsample(scale_factor=0.25, mode='bilinear')
        self.downsample3 = nn.Upsample(scale_factor=0.125, mode='bilinear')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsample5 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.conv_out_4 = nn.Conv2d(32,1,kernel_size=1)
        self.conv_out_3 = nn.Conv2d(32,1,kernel_size=1)
        self.conv_out_2 = nn.Conv2d(32,1,kernel_size=1)
        self.conv_out_1 = nn.Conv2d(32,1,kernel_size=1)
        self.conv_out_0 = nn.Conv2d(32,1,kernel_size=1)
       
        
        
    
    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'],strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        #self.depth_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        #print(f"Depth SwinTransformer loading pre_model ${pre_model}")
        
    def forward(self, x_rgb,x_t):
        # ************************* Encoder ***************************
        #part1
        r0,r1,r2,r3,r4 = self.rgb_swin(x_rgb)
        t0,t1,t2,t3,t4 = self.rgb_swin(x_t)
        


       
        r0 = self.conv_3x3_0_RGB(r0)
        r1 = self.conv_3x3_1_RGB(r1)
        r2 = self.conv_3x3_2_RGB(r2)
        r3 = self.conv_3x3_3_RGB(r3)
        r4 = self.conv_3x3_4_RGB(r4)

        t0 = self.conv_3x3_0_T(t0)
        t1 = self.conv_3x3_1_T(t1)
        t2 = self.conv_3x3_2_T(t2)
        t3 = self.conv_3x3_3_T(t3)
        t4 = self.conv_3x3_4_T(t4)

        rt0 = self.ra0(r0,t0)
        rt1 = self.ra1(r1,t1)
        rt2 = self.ra2(r2,t2)
        rt3 = self.ra3(r3,t3)
        rt4 = self.ra4(r4,t4)

        f0 = self.pfm0(rt0)
        f1 = self.pfm1(rt1)
        f2 = self.pfm2(rt2)
        f3 = self.se3(rt3)
        f4 = self.se4(rt4)

        d3 = self.conv_3x3_3_de(torch.cat((f3,self.upsample1(f4)),1))
        d2 = self.conv_3x3_2_de(torch.cat((f2,self.upsample1(d3)),1))
        d1 = self.conv_3x3_1_de(torch.cat((f1,self.upsample1(d2)),1))
        d0 = self.conv_3x3_0_de(torch.cat((f0,d1),1))

        Sal0 = self.conv_out_0(d0)
        Sal1 = self.conv_out_1(d1)
        Sal2 = self.conv_out_2(d2)
        Sal3 = self.conv_out_3(d3)
        Sal4 = self.conv_out_4(f4)



        #print(r0.size()) 128 96
        #print(r1.size()) 128, 96
        #print(r2.size()) 256, 48
        #print(r3.size()) 512 24
        #print(r4.size())  1024 12
        
        

       
        return self.upsample2(Sal0),self.upsample2(Sal1),self.upsample3(Sal2),self.upsample4(Sal3),self.upsample5(Sal4)

