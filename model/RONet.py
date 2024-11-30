 # coding=utf-8
from inspect import classify_class_attrs
from turtle import forward
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.conv import Conv2d
#from model.resattention import res_cbam
import torchvision.models as models
#from model.res2fg import res2net
import torch.nn.functional as F
import math
#from model.vgg import B2_VGG
from model.swinNet import SwinTransformer
class Gu(nn.Module):
    def __init__(self):
        super(Gu,self).__init__()
        
        
    def forward(self,x,bin):
        #print(bin[:,0,:,:].size())
        #print(x.size())
        a=torch.unsqueeze(bin[:,0,:,:],dim=1)
        b=torch.unsqueeze(bin[:,1,:,:],dim=1)
        #print(a.size())
        #print(x.size())
        out = x+x*b#+x*bin[:,2,:,:]

        #out = x1+self.conv_2(torch.cat((m2_c,m2_s),1))+self.conv_3(torch.cat((m3_c,m3_s),1))
        return out

class SelfAttention_1(nn.Module):
    def __init__(self,in_channels):
        super(SelfAttention_1, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels
        #max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        #bn = nn.BatchNorm2d

        self.g = nn.Conv2d(in_channels,in_channels,1)
        self.theta = nn.Conv2d(in_channels,in_channels,1)
        self.phi = nn.Conv2d(in_channels,in_channels,1)
    def forward(self,x3):

        batch_size = x3.size(0)

        g_x = self.g(x3).view(batch_size, self.inter_channels, -1)#[bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)#[b,wh,c]

        theta_x = self.theta(x3).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x3).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x3.size()[2:])
        out = y + x3
        
        return out

class GC(nn.Module):
    def __init__(self,in_channels):
        super(GC,self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_channels,in_channels),requires_grad=True)
        self.reset_para()
    def reset_para(self):
        stdv=1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
    def forward(self,x):
        batch_size = x.size(0)
        channel = x.size(1)
        #print(channel)
        g_x = x.view(batch_size, channel, -1)#[bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)#[b,wh,c]
        theta_x = x.view(batch_size, channel, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = x.view(batch_size, channel, -1)
        
        f = torch.matmul(theta_x, phi_x)

        adj = F.softmax(f, dim=-1)
        #print(g_x.size())
        #print(self.weight.size())
        support = torch.matmul(g_x,self.weight)
        y = torch.matmul(adj,support)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, channel, *x.size()[2:])
        return y+x

class SE(nn.Module):
    def __init__(self):
        super(SE,self).__init__()
        self.conv_1x1_0 = nn.Conv2d(128,64,kernel_size=1)
        self.conv_1x1_1 = nn.Conv2d(128,64,kernel_size=1)
        self.conv_1x1_4 = nn.Conv2d(128,64,kernel_size=1)
        self.CA_4_1 = ChannelAttention(128)
        self.CA_1_0 = ChannelAttention(128)
        self.conv_3x3_1 = nn.Conv2d(128,64,kernel_size=3,padding=1)
        self.conv_3x3_0 = nn.Conv2d(128,64,kernel_size=3,padding=1)
        self.conv_1x1_out = nn.Conv2d(64,1,kernel_size=1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear')
    def forward(self,x0,x1,x4):
        f0 = self.conv_1x1_0(x0)
        f1 = self.conv_1x1_1(x1)
        f4 = self.conv_1x1_4(x4)
        S1 = self.conv_3x3_1(self.CA_4_1(torch.cat((f1,self.upsample3(f4)),1)))
        S0 = self.conv_3x3_0(self.CA_1_0(torch.cat((f0,self.upsample1(S1)),1)))
        out = F.sigmoid(self.conv_1x1_out(S0))
        return out
class Encoder(nn.Module):
    def __init__(self,in_channels):
        super(Encoder,self).__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.conv_2 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
        
    def forward(self,x1,x2):
        s1 = self.sa1(x1)
        s2 = self.sa2(x2)
        f1 = x1*s2+x1
        f2 = x2*s1+x2
        ff = self.conv_2(torch.cat((f1,f2),1))
        out = self.ca(ff)+ff

        #out = x1+self.conv_2(torch.cat((m2_c,m2_s),1))+self.conv_3(torch.cat((m3_c,m3_s),1))
        return out
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.P4 = PFM(128)
        self.P3 = PFM(128)
        self.conv_3x3_4_3 = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_3_2 = nn.Conv2d(512,128,kernel_size=3,padding=1)
        self.conv_3x3_2_1 = nn.Conv2d(512,128,kernel_size=3,padding=1)
        self.conv_3x3_1_0 = nn.Conv2d(512,128,kernel_size=3,padding=1)

        self.conv_1x1_0 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_1x1_1 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_1x1_2 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_1x1_3 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_1x1_4 = nn.Conv2d(128,1,kernel_size=1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear')
    def forward(self,x0,x1,x2,x3,x4):
        p4 = self.P4(x4)
        p3 = self.P3(x3)
        f3 = self.conv_3x3_4_3(torch.cat((p3,self.upsample1(p4)),1))
        f2 = self.conv_3x3_3_2(torch.cat((x2,self.upsample1(f3),self.upsample1(p3),self.upsample2(p4)),1))
        f1 = self.conv_3x3_2_1(torch.cat((x1,self.upsample1(f2),self.upsample2(p3),self.upsample3(p4)),1))
        f0 = self.conv_3x3_1_0(torch.cat((x0,self.upsample1(f1),self.upsample3(p3),self.upsample4(p4)),1))

        #s0 = F.sigmoid(self.conv_1x1_0(f0))
        #s1 = F.sigmoid(self.conv_1x1_1(f1))
        #s2 = F.sigmoid(self.conv_1x1_2(f2))
        #s3 = F.sigmoid(self.conv_1x1_3(f3))
        #s4 = F.sigmoid(self.conv_1x1_4(p4))
        return f0,f1,f2,f3,p4
class MMI(nn.Module):
    def __init__(self,in_channels):
        super(MMI,self).__init__()
        self.CA_r = ChannelAttention_2(in_channels)
        self.SA_r = SpatialAttention()
        self.CA_d = ChannelAttention_2(in_channels)
        self.CA_t = ChannelAttention_2(in_channels)
        self.SA_dt = SpatialAttention()
        self.conv1 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(3*in_channels,in_channels,kernel_size=3,padding=1)
    def forward(self,r,d,t):
        sa_r = self.SA_r(self.CA_r(r)*r)
        d_r = d+sa_r*d
        t_r = t+sa_r*t
        dt = self.conv1(torch.cat((self.CA_d(d_r)*d_r,self.CA_t(t_r)*t_r),1))
        sa_dt = self.SA_dt(dt)
        out = r*sa_dt+dt*sa_dt
        return out
class FF(nn.Module):
    def __init__(self):
        super(FF,self).__init__()
        self.sa =SpatialAttention()
        self.conv1 = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv = nn.Conv2d(128,1,kernel_size=1)
    def forward(self,x1,x2):
              
        out = self.conv1(torch.cat((x1,x2),1))
        out = out * self.sa(out)+out
        S = F.sigmoid(self.conv(out))
        return out,S
class FFF(nn.Module):
    def __init__(self):
        super(FFF,self).__init__()
        self.sa =SpatialAttention()
        self.conv = nn.Conv2d(128,1,kernel_size=1)
        self.conv1 = nn.Conv2d(256+128,128,kernel_size=3,padding=1)
        
    def forward(self,x1,x2,x3,s):
        x1 = x1+x1*s
        x2 = x2+x2*s
        out = self.conv1(torch.cat((x1,x2,x3),1))
        out = out * self.sa(out)+out
        S = F.sigmoid(self.conv(out))
        return out,S
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
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class ChannelAttention_2(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_2, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 2, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 2, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
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
class SpatialAttention_m(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_m, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print(x.size())
        max_out = torch.mean(x, dim=1, keepdim=True)#dim=1是因为传入数据是多加了一维unsqueeze(0)
        x=max_out
        #print(x.size())
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
class Attention(nn.Module):
    def __init__(self,in_channels):
        super(Attention,self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
        self.ca = ChannelAttention_2(in_channels)
        self.sa = SpatialAttention()
        self.conv_out = nn.Conv2d(in_channels,1,kernel_size=1)
    def forward(self,x1,x2):
        a = self.conv(torch.cat((x1,x2),1))
        fa = self.ca(a)*a+a
        fs = self.sa(fa)*fa+fa
        return fs
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

        #self.atten0 = ChannelAttention_2(in_channels)
        #self.atten1 = ChannelAttention_2(in_channels)
        #self.atten2 = ChannelAttention_2(in_channels)

        self.conv = nn.Conv2d(4*in_channels,in_channels,kernel_size=3,padding=1)
        
    def forward(self,x):
        #x0,x1,x2,x3 = torch.split(x,x.size()[1]//4,dim=1)
        y0 = self.branch0(x)
        y1 = self.branch1(x+x*self.sa0(y0))
        y2 = self.branch2(x+x*self.sa1(y1))
        y3 = self.branch3(x+x*self.sa2(y2))

        y = self.conv(torch.cat((y0,y1,y2,y3),1))

        #out = y+y*self.sa(y)+x
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
class EDGE(nn.Module):
    def __init__(self,in_channels):
        super(EDGE,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv11 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv22 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv33 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)

        self.conv4 = nn.Conv2d(3*in_channels,in_channels,kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv_out = nn.Conv2d(in_channels,1,kernel_size=1)

        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
    def forward(self,x1,x2,x3):
        a = self.conv1(x1)
        b = self.conv2(x2)
        c = self.conv3(x3)

        b1 = self.conv22(torch.cat((b,c),1))
        a1 = self.conv33(torch.cat((a,b1),1))

        y3 = c*self.sa3(c)+c
        y2 = b1*self.sa2(b1)+b1
        y1 = a1*self.sa1(a1)+a1
        out = self.conv5(self.conv4(torch.cat((a,b,c),1)))
        #out = self.conv_out(out)
        
        return out
class Semantic(nn.Module):
    def __init__(self,in_channels):
        super(Semantic,self).__init__()    
        self.sa_max = SpatialAttention()
        self.sa_mea = SpatialAttention_m()
        self.conv = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
        self.sa_max_1 = SpatialAttention()
        self.sa_mea_1 = SpatialAttention_m()
        self.conv_1 = nn.Conv2d(2*in_channels,in_channels,kernel_size=3,padding=1)
    def forward(self,x):
        y0 = self.conv(torch.cat((self.sa_max(x)*x,self.sa_mea(x)*x),1))+x
        y1 = self.conv_1(torch.cat((self.sa_max_1(y0)*y0,self.sa_mea_1(y0)*y0),1))+y0
        return y1
class CBR(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CBR,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.Ba = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.relu(self.Ba(self.conv(x)))

        return out
class RONet(nn.Module):#输入三通道
    def __init__(self):
        super(RONet, self).__init__()
        self.r_swin= SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        #self.d_swin= SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        #self.t_swin= SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        self.gu_d0 = Gu()
        self.gu_d1 = Gu()
        self.gu_d2 = Gu()
        self.gu_d3 = Gu()
        self.gu_d4 = Gu()

        self.gu_t0 = Gu()
        self.gu_t1 = Gu()
        self.gu_t2 = Gu()
        self.gu_t3 = Gu()
        self.gu_t4 = Gu()

        self.gu_rd0 = Gu()
        self.gu_rd1 = Gu()
        self.gu_rd2 = Gu()
        self.gu_rd3 = Gu()
        self.gu_rd4 = Gu()

        self.gu_rt0 = Gu()
        self.gu_rt1 = Gu()
        self.gu_rt2 = Gu()
        self.gu_rt3 = Gu()
        self.gu_rt4 = Gu()

        self.conv_1x1_0_RGB = nn.Conv2d(64,128,kernel_size=1)
        self.conv_1x1_1_RGB = nn.Conv2d(128,128,kernel_size=1)
        self.conv_1x1_2_RGB = nn.Conv2d(256,128,kernel_size=1)
        self.conv_1x1_3_RGB = nn.Conv2d(512,128,kernel_size=1)
        self.conv_1x1_4_RGB = nn.Conv2d(1024,128,kernel_size=1)

        self.conv_1x1_0_RGBt = nn.Conv2d(64,128,kernel_size=1)
        self.conv_1x1_1_RGBt = nn.Conv2d(128,128,kernel_size=1)
        self.conv_1x1_2_RGBt = nn.Conv2d(256,128,kernel_size=1)
        self.conv_1x1_3_RGBt = nn.Conv2d(512,128,kernel_size=1)
        self.conv_1x1_4_RGBt = nn.Conv2d(1024,128,kernel_size=1)
        
        self.conv_1x1_0_D = nn.Conv2d(64,128,kernel_size=1)
        self.conv_1x1_1_D = nn.Conv2d(128,128,kernel_size=1)
        self.conv_1x1_2_D = nn.Conv2d(256,128,kernel_size=1)
        self.conv_1x1_3_D = nn.Conv2d(512,128,kernel_size=1)
        self.conv_1x1_4_D = nn.Conv2d(1024,128,kernel_size=1)

        self.conv_1x1_0_T = nn.Conv2d(64,128,kernel_size=1)
        self.conv_1x1_1_T = nn.Conv2d(128,128,kernel_size=1)
        self.conv_1x1_2_T = nn.Conv2d(256,128,kernel_size=1)
        self.conv_1x1_3_T = nn.Conv2d(512,128,kernel_size=1)
        self.conv_1x1_4_T = nn.Conv2d(1024,128,kernel_size=1)

        self.enrd0 = Encoder(128)
        self.enrd1 = Encoder(128)
        self.enrd2 = Encoder(128)
        self.enrd3 = Encoder(128)
        self.enrd4 = Encoder(128)

        self.enrt0 = Encoder(128)
        self.enrt1 = Encoder(128)
        self.enrt2 = Encoder(128)
        self.enrt3 = Encoder(128)
        self.enrt4 = Encoder(128)


        
        

        
        # ************************* Decoder ***************************
        #self.Se_D = SE()
        #self.Se_T = SE()
        self.conv_rd1 = nn.Conv2d(256,128,kernel_size=1)
        self.conv_rd0 = nn.Conv2d(256,128,kernel_size=1)

        self.conv_rt1 = nn.Conv2d(256,128,kernel_size=1)
        self.conv_rt0 = nn.Conv2d(256,128,kernel_size=1)

        self.conv_dt1 = nn.Conv2d(256,128,kernel_size=1)
        self.conv_dt0 = nn.Conv2d(256,128,kernel_size=1)

        self.conv_rdt1 = nn.Conv2d(384,128,kernel_size=1)
        self.conv_rdt0 = nn.Conv2d(384,128,kernel_size=1)

        self.conv4_1 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv3_1 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv2_1 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv1_1 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv0_1 = nn.Conv2d(384,128,kernel_size=3,padding=1)

        self.conv4_2 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv3_2 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv2_2 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv1_2 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv0_2 = nn.Conv2d(384,128,kernel_size=3,padding=1)

        #self.edge = EDGE(128)
        
        self.pfm_rd0 = PFM(128)
        self.pfm_rd1 = PFM(128)
        self.pfm_rd2 = PFM(128)
        self.pfm_rd3 = PFM(128)
        self.pfm_rd4 = PFM(128)

        self.pfm_rt0 = PFM(128)
        self.pfm_rt1 = PFM(128)
        self.pfm_rt2 = PFM(128)
        self.pfm_rt3 = PFM(128)
        self.pfm_rt4 = PFM(128)
        
        self.an_rd0 = Attention(128)
        self.an_rd1 = Attention(128)
        self.an_rd2 = Attention(128)
        self.an_rd3 = Attention(128)

        self.an_rt0 = Attention(128)
        self.an_rt1 = Attention(128)
        self.an_rt2 = Attention(128)
        self.an_rt3 = Attention(128)


        self.conv_3x3_0_RGB = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_1_RGB = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_2_RGB = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_3_RGB = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_4_RGB = nn.Conv2d(256,128,kernel_size=3,padding=1)

        self.conv_3x3_0_T = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_1_T = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_2_T = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_3_T = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_3x3_4_T = nn.Conv2d(256,128,kernel_size=3,padding=1)

        #self.decoder_RGB = Decoder()
        #self.decoder_T = Decoder()

        #self.loc = location(128)
        #self.Fuse = FF()
        #self.fff_4 = FF(128)
        #self.fff_3 = FFF(128)
        #self.fff_2 = FFF(128)
        #self.fff_1 = FFF(128)
        #self.fff_0 = FFF(128)
        self.conv_a2 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv_a1 = nn.Conv2d(512,128,kernel_size=3,padding=1)
        self.conv_a0 = nn.Conv2d(640,128,kernel_size=3,padding=1)
        self.se4_attention = Semantic(128)
        self.se3_attention = Semantic(128)
        self.se2_attention = Semantic(128)
        self.se1_attention = Semantic(128)
        self.se0_attention = Semantic(128)
        
        self.fu4 = FF()
        self.fu3 = FFF()
        self.fu2 = FFF()
        self.fu1 = FFF()
        self.fu0 = FFF()
        self.conv_s3 = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.conv_s2 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv_s1 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.conv_s0 = nn.Conv2d(384,128,kernel_size=3,padding=1)
        self.at2 = Attention(128)
        self.at1 = Attention(128)
        self.at0 = Attention(128)
        self.conv_2 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.conv_1 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.conv_0 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.conv_e = nn.Conv2d(128,1,kernel_size=1)

        # ************************* Feature Map Upsample ***************************
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear')
        self.downsample2 = nn.Upsample(scale_factor=0.25, mode='bilinear')
        self.downsample3 = nn.Upsample(scale_factor=0.125, mode='bilinear')
        self.downsample4 = nn.Upsample(scale_factor=0.0625, mode='bilinear')
        self.downsample5 = nn.Upsample(scale_factor=0.03125, mode='bilinear')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsample5 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.conv_out_4 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_out_3 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_out_2 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_out_1 = nn.Conv2d(128,1,kernel_size=1)
        self.conv_out_0 = nn.Conv2d(128,1,kernel_size=1)
    def load_pre(self, pre_model):
        self.r_swin.load_state_dict(torch.load(pre_model)['model'],strict=False)
        #self.d_swin.load_state_dict(torch.load(pre_model)['model'],strict=False)
        #self.t_swin.load_state_dict(torch.load(pre_model)['model'],strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")

    def forward(self, x_rgb,x_d,x_t,d_bin,t_bin):
        # ************************* Encoder ***************************
        #D>>R
        #print(x_rgb.size())
        r0, r1, r2, r3, r4 = self.r_swin(x_rgb)#128,256,512,1024
        d0, d1, d2, d3, d4 = self.r_swin(x_d)
        t0, t1, t2, t3, t4 = self.r_swin(x_t)
        

        
        #print(sx_d.size())
        #print(d_bin.size())
        #sx_d = self.gu_d0(sx_d,d_bin)
        s1_d = self.gu_d1(d1,self.downsample2(d_bin))
        s2_d = self.gu_d2(d2,self.downsample3(d_bin))
        s3_d = self.gu_d3(d3,self.downsample4(d_bin))
        s4_d = self.gu_d4(d4,self.downsample5(d_bin))

        #sx_t = self.gu_t0(sx_t,t_bin)
        s1_t = self.gu_t1(t1,self.downsample2(t_bin))
        s2_t = self.gu_t2(t2,self.downsample3(t_bin))
        s3_t = self.gu_t3(t3,self.downsample4(t_bin))
        s4_t = self.gu_t4(t4,self.downsample5(t_bin))

        #sx_rd = self.gu_rd0(sx_r,d_bin)
        s1_rd = self.gu_rd1(r1,self.downsample2(d_bin))
        s2_rd = self.gu_rd2(r2,self.downsample3(d_bin))
        s3_rd = self.gu_rd3(r3,self.downsample4(d_bin))
        s4_rd = self.gu_rd4(r4,self.downsample5(d_bin))

        #sx_rt = self.gu_rt0(sx_r,t_bin)
        s1_rt = self.gu_rt1(r1,self.downsample2(t_bin))
        s2_rt = self.gu_rt2(r2,self.downsample3(t_bin))
        s3_rt = self.gu_rt3(r3,self.downsample4(t_bin))
        s4_rt = self.gu_rt4(r4,self.downsample5(t_bin))

        #s0_d = self.conv_1x1_0_D(sx_d)
        s1_d = self.conv_1x1_1_D(s1_d)
        s2_d = self.conv_1x1_2_D(s2_d)
        s3_d = self.conv_1x1_3_D(s3_d)
        s4_d = self.conv_1x1_4_D(s4_d)

        #s0_t = self.conv_1x1_0_T(sx_t)
        s1_t = self.conv_1x1_1_T(s1_t)
        s2_t = self.conv_1x1_2_T(s2_t)
        s3_t = self.conv_1x1_3_T(s3_t)
        s4_t = self.conv_1x1_4_T(s4_t)


        #s0_rd = self.conv_1x1_0_RGB(sx_rd)
        s1_rd = self.conv_1x1_1_RGB(s1_rd)
        s2_rd = self.conv_1x1_2_RGB(s2_rd)
        s3_rd = self.conv_1x1_3_RGB(s3_rd)
        s4_rd = self.conv_1x1_4_RGB(s4_rd)

        #s0_rt = self.conv_1x1_0_RGBt(sx_rt)
        s1_rt = self.conv_1x1_1_RGBt(s1_rt)
        s2_rt = self.conv_1x1_2_RGBt(s2_rt)
        s3_rt = self.conv_1x1_3_RGBt(s3_rt)
        s4_rt = self.conv_1x1_4_RGBt(s4_rt)
        ##################################
        
        #frd0 = self.enrd0(s0_rd,s0_d)
        frd1 = self.enrd1(s1_rd,s1_d)
        frd2 = self.enrd2(s2_rd,s2_d)
        frd3 = self.enrd3(s3_rd,s3_d)
        frd4 = self.enrd4(s4_rd,s4_d)

        #frt0 = self.enrt0(s0_rt,s0_t)
        frt1 = self.enrt1(s1_rt,s1_t)
        frt2 = self.enrt2(s2_rt,s2_t)
        frt3 = self.enrt3(s3_rt,s3_t)
        frt4 = self.enrt4(s4_rt,s4_t)
        ###1
        
        #prd0 = self.pfm_rd0(frd0)
        prd1 = self.pfm_rd1(frd1)
        prd2 = self.pfm_rd2(frd2)
        prd3 = self.pfm_rd3(frd3)
        prd4 = self.pfm_rd4(frd4)

        #prt0 = self.pfm_rt0(frt0)
        prt1 = self.pfm_rt1(frt1)
        prt2 = self.pfm_rt2(frt2)
        prt3 = self.pfm_rt3(frt3)
        prt4 = self.pfm_rt4(frt4)

        ard4 = prd4
        ard3 = self.an_rd3(prd3,self.upsample1(ard4))
        ard2 = self.an_rd2(prd2,self.upsample1(ard3))
        ard1 = self.an_rd1(prd1,self.upsample1(ard2))
        #ard0 = self.an_rd0(prd0,self.upsample1(ard1))

        art4 = prt4
        art3 = self.an_rt3(prt3,self.upsample1(art4))
        art2 = self.an_rt2(prt2,self.upsample1(art3))
        art1 = self.an_rt1(prt1,self.upsample1(art2))
        #rt0 = self.an_rt0(prt0,self.upsample1(art1))

        ff4,Sal4 = self.fu4(ard4,art4)
        ff3,Sal3 = self.fu3(ard3,art3,self.upsample1(ff4),self.upsample1(Sal4))
        ff2,Sal2 = self.fu2(ard2,art2,self.upsample1(ff3),self.upsample1(Sal3))
        ff1,Sal1 = self.fu1(ard1,art1,self.upsample1(ff2),self.upsample1(Sal2))
        #ff0,Sal0 = self.fu0(ard0,art0,self.upsample1(ff1),self.upsample1(Sal1))

        
        return self.upsample2(Sal1),self.upsample3(Sal2),self.upsample4(Sal3),self.upsample5(Sal4)#,G_d,G_t


