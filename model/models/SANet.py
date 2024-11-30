# import sys
# sys.path.append('./')
# print(sys.path)
import torch
from torch import nn
from torch.nn import functional as F
# from torchvision.models import vgg16,resnet34
# from models.vgg import B2_VGG
from models.swinNet import SwinTransformer
import copy
import inspect
from functools import reduce
import cv2
from torch.nn import BatchNorm2d as bn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# resnet_2D = resnet50(pretrained=True)
# resnet_3D = resnet50(pretrained=True)


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

class BasicConv2d_relu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d_relu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.bn(x))
        return x






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

class SE(nn.Module):
    def __init__(self):
        super(SE,self).__init__()
        # self.conv_1x1_0 = nn.Conv2d(64,64,kernel_size=1)
        # self.conv_1x1_1 = nn.Conv2d(64,64,kernel_size=1)
        # self.conv_1x1_4 = nn.Conv2d(512,64,kernel_size=1)
        self.CA_4_1 = ChannelAttention(32*2)
        self.CA_1_0 = ChannelAttention(32*2)
        self.conv_3x3_1 = nn.Conv2d(64,32,kernel_size=3,padding=1)
        self.conv_3x3_0 = nn.Conv2d(64,32,kernel_size=3,padding=1)
        self.conv_1x1_out = nn.Conv2d(32,1,kernel_size=1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear')
    def forward(self,f0,f1,f4):
    
        S1 = self.conv_3x3_1(self.CA_4_1(torch.cat((f1,self.upsample3(f4)),1)))
        S0 = self.conv_3x3_0(self.CA_1_0(torch.cat((f0,S1),1)))
        out = F.sigmoid(self.conv_1x1_out(S0))
        return out


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

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.P4 = PFM(32)
        self.P3 = PFM(32)
        self.conv_3x3_4_3 = nn.Conv2d(32*2,32,kernel_size=3,padding=1)
        self.conv_3x3_3_2 = nn.Conv2d(32*4,32,kernel_size=3,padding=1)
        self.conv_3x3_2_1 = nn.Conv2d(32*4,32,kernel_size=3,padding=1)
        self.conv_3x3_1_0 = nn.Conv2d(32*4,32,kernel_size=3,padding=1)

        #self.conv_1x1_0 = nn.Conv2d(64,1,kernel_size=1)
        #self.conv_1x1_1 = nn.Conv2d(64,1,kernel_size=1)
        #self.conv_1x1_2 = nn.Conv2d(128,1,kernel_size=1)
        #self.conv_1x1_3 = nn.Conv2d(256,1,kernel_size=1)
        #self.conv_1x1_4 = nn.Conv2d(512,1,kernel_size=1)

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
        f0 = self.conv_3x3_1_0(torch.cat((x0,f1,self.upsample2(p3),self.upsample3(p4)),1))

        #s0 = F.sigmoid(self.conv_1x1_0(f0))
        #s1 = F.sigmoid(self.conv_1x1_1(f1))
        #s2 = F.sigmoid(self.conv_1x1_2(f2))
        #s3 = F.sigmoid(self.conv_1x1_3(f3))
        #s4 = F.sigmoid(self.conv_1x1_4(p4))
        return f0,f1,f2,f3,p4

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

class FF(nn.Module):
    def __init__(self):
        super(FF,self).__init__()
        self.conv1 = nn.Conv2d(64,32,kernel_size=3,padding=1)
        self.CA = ChannelAttention_1(32)
        #self.conv2 = nn.Conv2d(64,1,kernel_size=1)
    def forward(self,x1,x2):
        c = self.conv1(torch.cat((x1,x2),1))
        v = self.CA(c)
        #y1 = x1*v
        #y2 = x2*v
        out = c*v+c
        return out
class FFF(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(FFF,self).__init__()
        self.conv_1 = nn.Conv2d(2*in_channels,out_channels,kernel_size=3,padding=1)
        self.conv_2 = nn.Conv2d(2*in_channels,out_channels,kernel_size=3,padding=1)
        self.CA = ChannelAttention_1(out_channels)
        self.SA = SpatialAttention()
    def forward(self,x1,x2,x3):
        c1 = self.conv_1(torch.cat((x1,x2),1))
        c2 = self.conv_2(torch.cat((x2,x3),1))
        v = self.CA(c1)
        w = self.SA(c2)
        out = c1*v+c2*w+c1+c2
        return out


class SANet(nn.Module):
  def __init__(self):
    super(SANet, self).__init__()

    channel_decoder = 32

    
    # encoder
    self.rgb_swin= SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
    # self.t_swin= SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])

    self.Se_RGB = SE()
    self.Se_T = SE()

    self.rfb5= BasicConv2d_relu(512*2,channel_decoder,3,1,1)
    self.rfb4= BasicConv2d_relu(256*2,channel_decoder,3,1,1)
    self.rfb3= BasicConv2d_relu(128*2,channel_decoder,3,1,1)
    self.rfb2= BasicConv2d_relu(64*2,channel_decoder,3,1,1)
    self.rfb1= BasicConv2d_relu(64*2,channel_decoder,3,1,1)

    self.rfb5_t= BasicConv2d_relu(512*2,channel_decoder,3,1,1)
    self.rfb4_t= BasicConv2d_relu(256*2,channel_decoder,3,1,1)
    self.rfb3_t= BasicConv2d_relu(128*2,channel_decoder,3,1,1)
    self.rfb2_t= BasicConv2d_relu(64*2,channel_decoder,3,1,1)
    self.rfb1_t= BasicConv2d_relu(64*2,channel_decoder,3,1,1)



   
    self.conv_3x3_0_RGB = nn.Conv2d(64,32,kernel_size=3,padding=1)
    self.conv_3x3_1_RGB = nn.Conv2d(64,32,kernel_size=3,padding=1)
    self.conv_3x3_2_RGB = nn.Conv2d(64,32,kernel_size=3,padding=1)
    self.conv_3x3_3_RGB = nn.Conv2d(64,32,kernel_size=3,padding=1)
    self.conv_3x3_4_RGB = nn.Conv2d(64,32,kernel_size=3,padding=1)
    
    self.conv_3x3_0_T = nn.Conv2d(64,32,kernel_size=3,padding=1)
    self.conv_3x3_1_T = nn.Conv2d(64,32,kernel_size=3,padding=1)
    self.conv_3x3_2_T = nn.Conv2d(64,32,kernel_size=3,padding=1)
    self.conv_3x3_3_T = nn.Conv2d(64,32,kernel_size=3,padding=1)
    self.conv_3x3_4_T = nn.Conv2d(64,32,kernel_size=3,padding=1)

    self.decoder_RGB = Decoder()
    self.decoder_T = Decoder()
    
    self.conv0= BasicConv2d_relu(64*2,channel_decoder,3,1,1)
    self.conv1= BasicConv2d_relu(64*2,channel_decoder,3,1,1)
    self.conv2= BasicConv2d_relu(128*2,channel_decoder,3,1,1)
    self.conv3= BasicConv2d_relu(256*2,channel_decoder,3,1,1)
    self.conv4= BasicConv2d_relu(512*2,channel_decoder,3,1,1)
    self.conv5= BasicConv2d_relu(64*2,channel_decoder,3,1,1)
    self.conv6= BasicConv2d_relu(64*2,channel_decoder,3,1,1)
    self.conv7= BasicConv2d_relu(128*2,channel_decoder,3,1,1)
    self.conv8= BasicConv2d_relu(256*2,channel_decoder,3,1,1)
    self.conv9= BasicConv2d_relu(512*2,channel_decoder,3,1,1)
    
    # self.dec_fus1= nn.Sequential(    BasicConv2d_relu(channel_decoder*2,channel_decoder,3,1,1),)
    # self.fuse_final= nn.Sequential(    BasicConv2d_relu(channel_decoder,channel_decoder,3,1,1),)

    # self.conv_up1= BasicConv2d_relu(channel_decoder,channel_decoder,3,1,1)
    # self.conv_up2= BasicConv2d_relu(channel_decoder,channel_decoder,3,1,1)
    self.Fuse = FF()
    self.fff_3 = FFF(32,32)
    self.fff_2 = FFF(32,32)
    self.fff_1 = FFF(32,32)
    self.fff_0 = FFF(32,32)

    self.conv_out_4 = nn.Conv2d(32,1,kernel_size=1)
    self.conv_out_3 = nn.Conv2d(32,1,kernel_size=1)
    self.conv_out_2 = nn.Conv2d(32,1,kernel_size=1)
    self.conv_out_1 = nn.Conv2d(32,1,kernel_size=1)
    self.conv_out_0 = nn.Conv2d(32,1,kernel_size=1)

    self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
    self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
    self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

    # if self.training: self.initialize_weights()

  def load_pre(self, pre_model):
    self.rgb_swin.load_state_dict(torch.load(pre_model)['model'],strict=False)
    print(f"RGB SwinTransformer loading pre_model ${pre_model}")
    # self.t_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
    print(f"Depth SwinTransformer loading pre_model ${pre_model}")
  def forward(self, x_rgb, x_t):
    # guidance of high-level encoding of focus stacking
    f0_rgb, f1_rgb, f2_rgb, f3_rgb, f4_rgb = self.rgb_swin(x_rgb)
    f0_rgb, f1_rgb, f2_rgb, f3_rgb, f4_rgb= self.rfb1(f0_rgb), self.rfb2(f1_rgb), self.rfb3(f2_rgb), self.rfb4(f3_rgb), self.rfb5(f4_rgb)



    G_rgb = self.Se_RGB(f0_rgb,f1_rgb,f4_rgb)

    S_t = torch.mul(x_t,self.upsample4(G_rgb))
    #print(x_t.size())
    #print(S_t.size())

    s0_t, s1_t, s2_t, s3_t, s4_t = self.rgb_swin(S_t)
    s0_t, s1_t, s2_t, s3_t, s4_t= self.rfb1_t(s0_t), self.rfb2_t(s1_t), self.rfb3_t(s2_t), self.rfb4_t(s3_t), self.rfb5_t(s4_t)

    R0 = self.conv_3x3_0_RGB(torch.cat((f0_rgb,s0_t),1))
    R1 = self.conv_3x3_1_RGB(torch.cat((f1_rgb,s1_t),1))
    R2 = self.conv_3x3_2_RGB(torch.cat((f2_rgb,s2_t),1))
    R3 = self.conv_3x3_3_RGB(torch.cat((f3_rgb,s3_t),1))
    R4 = self.conv_3x3_4_RGB(torch.cat((f4_rgb,s4_t),1))

    f0_R,f1_R,f2_R,f3_R,f4_R = self.decoder_RGB(R0,R1,R2,R3,R4)

    #PART2
    #T Extract Features
    f0_t, f1_t, f2_t, f3_t, f4_t = self.rgb_swin(x_t)
    f0_t, f1_t, f2_t, f3_t, f4_t= self.conv0(f0_t), self.conv1(f1_t), self.conv2(f2_t), self.conv3(f3_t), self.conv4(f4_t)


    G_t = self.Se_T(f0_t,f1_t,f4_t)
    S_r = torch.mul(x_rgb,self.upsample4(G_t))

    
    s0_r, s1_r, s2_r, s3_r, s4_r = self.rgb_swin(S_r)
    s0_r, s1_r, s2_r, s3_r, s4_r= self.conv5(s0_r), self.conv6(s1_r), self.conv7(s2_r), self.conv8(s3_r), self.conv9(s4_r)

    T0 = self.conv_3x3_0_T(torch.cat((f0_t,s0_r),1))
    T1 = self.conv_3x3_1_T(torch.cat((f1_t,s1_r),1))
    T2 = self.conv_3x3_2_T(torch.cat((f2_t,s2_r),1))
    T3 = self.conv_3x3_3_T(torch.cat((f3_t,s3_r),1))
    T4 = self.conv_3x3_4_T(torch.cat((f4_t,s4_r),1))

    f0_T,f1_T,f2_T,f3_T,f4_T = self.decoder_T(T0,T1,T2,T3,T4)

    F4 = self.Fuse(f4_R,f4_T)
    F3 = self.fff_3(f3_R,self.upsample2(F4),f3_T)
    F2 = self.fff_2(f2_R,self.upsample2(F3),f2_T)
    F1 = self.fff_1(f1_R,self.upsample2(F2),f1_T)
    F0 = self.fff_0(f0_R,F1,f0_T)

    Sal0 = self.conv_out_0(F0)
    Sal1 = self.conv_out_1(F1)
    Sal2 = self.conv_out_2(F2)
    Sal3 = self.conv_out_3(F3)
    Sal4 = self.conv_out_4(F4)
    return self.upsample4(Sal0),self.upsample4(Sal1),self.upsample8(Sal2),self.upsample16(Sal3),self.upsample32(Sal4)

    

  #self.deconv1 = self._make_transpose(TransBasicBlock, channel_decoder,channel_decoder, 3, stride=2)
  def _make_transpose(self, block, inplanes, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        #先加2个block，且不变化通道
        for i in range(1, blocks):
            layers.append(block(inplanes, inplanes))

        layers.append(block(inplanes, planes, stride, upsample))
        inplanes = planes

        return nn.Sequential(*layers)
  def initialize_weights(self):
    # st= vgg16(pretrained=True).state_dict()
    # st2={}
    # for key in st.keys():
    #     st2['base.'+key]=st[key]
    # self.rgb_net.load_state_dict(st2)
    # self.t_net.load_state_dict(st2)
    # print('loading pretrained model success!')
    pass

if __name__ == '__main__':
    from thop import profile
    a= SANet()
    # torch.save(a.state_dict(),'./test0224.pth')
    # for name, module in a.encoder_fs.named_children():
    #     print(name, module)
    # model = resnet50()
    input1 = torch.randn(1, 3, 224, 224)
    input2 = torch.randn(1, 3, 224, 224)
    macs, params = profile(a, inputs=(input1,input2))
    print(macs,params)