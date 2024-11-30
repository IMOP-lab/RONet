# RONet:  Recurrent Optimization Network for RGB-T Salient Object Detection

[Project page](https://github.com/IMOP-lab/RONet) | [Our laboratory home page](https://github.com/IMOP-lab) 

Our paper has been accepted by IEEE Transactions on Medical Imaging!

by Yaoqi Sun, Zhaoyang Xu, Zhao Huang, Gaopeng Huang, Bin Wan, Haibing Yin, Jin Liu, Zhiwen Zheng, Xiaoshuai Zhang, and
Xingru Huang

Hangzhou Dianzi University IMOP-lab

<div align=center>
  <img src="https://github.com/IMOP-lab/RONet/blob/main/figures/stru.png">
</div>
<p align=center>
  Figure 1: The architecture of the proposed Recurrent Optimization Network (RONet) for RGB-T salient object detection, encompassing the Recurrent Optimization (RO) module for multi-modal feature refinement, the Multi-scale Semantic (MS) module for spatial information extraction, and the Detail Enhancement (DE) module for fine-grained feature enhancement, collectively facilitating high-quality salient object detection across multi-modal inputs.
</p>

## Installation

We trained and tested our RONet on three public RGB-T datasets: VT5000 , VT1000 , and VT821 . The VT5000 dataset, which is large-scale, contains 5,000 images of day scenes, equally divided into 2,500 training images and 2,500 test images. VT1000 and VT821 datasets include 1,000 and 821 registered RGB-T images, respectively, which are used solely for testing. Our proposed network is implemented using the PyTorch framework and was executed on a computing station equipped with a single NVIDIA GeForce RTX 4080 GPU. In this study, the parameters of the dual-modal backbone were initialized using the pre-trained SwinNet model \cite{liu2021swinnet}, while other parameters were randomly initialized. Prior to training, both training and test images were resized to 384 × 384 to minimize the utilization of computing resources. 

# Question

if you have any questions, please contact 'gaopeng.huang@hdu.edu.cn'
