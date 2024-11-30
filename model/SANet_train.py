import os
import torch
import torch.nn.functional as F
import torch.nn as nn
# import sys
# sys.path.append('./models')
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from data import get_loader
from utils import clip_gradient, adjust_lr
# from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
from utils import print_network
from models.SANet import SANet
from models.RONet import RONet
from utils import hybrid_e_loss
from torch.optim.lr_scheduler import MultiStepLR
import pytorch_ssim
import pytorch_iou
from data import Test_dataset
import torch.utils.data as data
from boundary_loss import BoundaryLoss


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
#set the device for training
# if opt.gpu_id == '0':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     print('USE GPU 0')
# elif opt.gpu_id == '1':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#     print('USE GPU 1')
cudnn.benchmark = True

#build the model
# model = torch.nn.DataParallel(SANet())
model = RONet(3)
model.load_pre('/home/wanbin/wanbin/pre_model/swin_base_patch4_window12_384_22k.pth')
# model.load_state_dict(torch.load('/home/lvchengtao/code/RGBT-395/BBSNet_cpts/BBSNet_epoch_93_1.pth'))
print_network(model, 'RONet')
# if(opt.load is not None):
#     model.load_state_dict(torch.load(opt.load))
#     print('load model from ', opt.load)
model= nn.DataParallel(model).cuda()
# model.load_state_dict(torch.load(''))
# params = model.parameters()
params1, params2=[],[]
for name, param in model.named_parameters():
    if check_keywords_in_name(name, ('rgb_swin','t_swin')):
        params1.append(param)
    else:
        params2.append(param)
# print(params1,'\n',params2)

# optimizer = torch.optim.SGD(params, opt.lr, weight_decay=0.0005,momentum=0.9)
optimizer = torch.optim.Adam([{'params':params1, 'lr': opt.lr*0.1},{'params':params2}], opt.lr)
# optimizer = torch.optim.SGD([{'params': params1}, {'params': params2}], lr=opt.lr, momentum=opt.momentum,
#                                 weight_decay=opt.decay_rate, nesterov=True)


milestones=[60,60+12,60+12*2,60+12*3,60+12*4]
scheduler_focal = MultiStepLR(optimizer, milestones, gamma=0.7, last_epoch=-1)

#set the path
image_root = '/data0/wanbin/wanbin/new_data/rgbt/VT5000/VT5000_clear/Train/RGB/'
gt_root = '/data0/wanbin/wanbin/new_data/rgbt/VT5000/VT5000_clear/Train/GT/'
depth_root= '/data0/wanbin/wanbin/new_data/rgbt/VT5000/VT5000_clear/Train/T/'
save_path='./BBSNet_cpts/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data
print('load data...')
train_loader = get_loader(image_root, gt_root, depth_root, batchsize= 4, trainsize= 384)
# test_loader = test_dataset(test_image_root, test_gt_root,test_depth_root, opt.trainsize)
total_step = len(train_loader)

# dataset_path= '/home/lvchengtao/dataset/rgbt/VT5000/VT5000_clear/Test/'
# image_root = dataset_path + '/RGB/'
# gt_root = dataset_path +'/GT/'
# depth_root=dataset_path +'/T/'
# test_dataset=Test_dataset(image_root, gt_root,depth_root, 384)
# batch= 32
# test_loader= data.DataLoader(dataset=test_dataset,
#     batch_size=batch,
#     shuffle=False,
#     num_workers=4,
#     pin_memory=True)

#set loss function
def floss(prediction, target, beta=0.3, log_like=False):
    prediction= torch.sigmoid(prediction)
    EPS = 1e-10
    N = prediction.size(0)
    TP = (prediction * target).view(N, -1).sum(dim=1)
    H = beta * target.view(N, -1).sum(dim=1) + prediction.view(N, -1).sum(dim=1)
    fmeasure = (1 + beta) * TP / (H + EPS)
    if log_like:
        flosss = -torch.log(fmeasure)
    else:
        flosss  = (1 - fmeasure)
    flosss=torch.mean(flosss)
    return flosss
CE= torch.nn.BCEWithLogitsLoss()
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)
def ssim_iou_loss(pred,target):

    pred= torch.sigmoid(pred)
    ssim_out = 1 - ssim_loss(pred,target)
    iou_out = iou_loss(pred,target)

    loss = ssim_out + iou_out

    return loss
def only_iou_loss(pred,target):

    pred= torch.sigmoid(pred)
    # ssim_out = 1 - ssim_loss(pred,target)
    iou_out = iou_loss(pred,target)

    loss = iou_out

    return loss


step = 0

best_mae = 1
best_epoch = 1
bd_loss= BoundaryLoss()

#train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, depths) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
          
            depths = depths.cuda()

            pred1,pred2,pred3,pred4,pred5= model(images,depths)
            loss1= CE(pred1, gts)+ only_iou_loss(pred1, gts)+floss(pred1, gts)
            loss2= CE(pred2, gts)+ only_iou_loss(pred2, gts)+floss(pred2, gts)
            loss3= CE(pred3, gts)+ only_iou_loss(pred3, gts)+floss(pred3, gts)
            loss4= CE(pred4, gts)+ only_iou_loss(pred4, gts)+floss(pred4, gts)
            loss5= CE(pred5, gts)+ only_iou_loss(pred5, gts)+floss(pred5, gts)
            #loss6= CE(pred6, gts)+ only_iou_loss(pred6, gts)+floss(pred6, gts)
            #loss7= CE(pred7, gts)+ only_iou_loss(pred7, gts)+floss(pred7, gts)
            #loss8= CE(pred8, gts)+ only_iou_loss(pred8, gts)+floss(pred8, gts)
            # loss6= CE(pred6, gts)+ only_iou_loss(pred6, gts)
            # loss7= CE(pred7, gts)+ only_iou_loss(pred7, gts)

            # loss_edge= bd_loss(pred2, gts)+ bd_loss(pred3, gts)+ bd_loss(pred4, gts)
            # loss11= ssim_iou_loss(pred1, gts)

            loss = loss1+loss2+loss3*(1./4.)+loss4*(1./16.)+loss5*(1./64.)
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 10 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, Loss1: {:.4f}, loss_edge: {:.4f},'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss1.data, 0))

        if (epoch) % 2 == 0 and (epoch)>60:
            torch.save(model.state_dict(), save_path+'BBSNet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')
        print('save checkpoints successfully!')
        raise

#test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        mae_num = 0
        for i, (image, gt, depth, name) in enumerate(test_loader):
            # print('batch {:4d}'.format(i))
            image = image.cuda()
            depth = depth.cuda()
            # gt= F.adaptive_avg_pool2d(gt,224)

            gt = np.asarray(gt, np.float32)
            gt = gt > 0.5

            bz= image.shape[0]
            _,pred,_,_,_,_,_ = model(image, depth)
            # pred= F.adaptive_avg_pool2d(pred,224)

            for j in range(bz):
                res2= pred[j]
                gt2= gt[j].squeeze()
                res2 = res2.sigmoid().data.cpu().numpy().squeeze()
                res2 = np.round((res2 - res2.min()) / (res2.max() - res2.min() + 1e-8)* 255.0)/ 255.0
                mae_sum += np.mean(np.abs(res2-gt2))
                mae_num += 1

        mae = mae_sum/ mae_num

        # print('##TEST##:Epoch: {}   MAE: {}'.format(epoch,mae))
        
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path+'BBSNet_epoch_best.pth')
                print('##SAVE##:bestEpoch: {}   bestMAE: {}'.format(best_epoch,best_mae))
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
 
if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch+1):
        # optimizer.param_groups[0]['lr'] = (1 - abs((epoch) / (opt.epoch) * 2 - 1)) * opt.lr * 0.1
        # optimizer.param_groups[1]['lr'] = (1 - abs((epoch) / (opt.epoch) * 2 - 1)) * opt.lr
        # cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        print('learning_rate', optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
        train(train_loader, model, optimizer, epoch, save_path)
        # test(test_loader, model, epoch, save_path)
        scheduler_focal.step()
    
    # optimizer = torch.optim.Adam([{'params':params1, 'lr': 1e-4* 0.1*0.8},{'params':params2}], 1e-4*0.8)
    # milestones=[30,30+6,30+6*2,30+6*3,30+6*4]
    # scheduler_focal = MultiStepLR(optimizer, milestones, gamma=0.7, last_epoch=-1)
    # for epoch in range(121, 180 + 1):
    #     print('learning_rate', optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
    #     train(train_loader, model, optimizer, epoch, save_path)
    #     # test(test_loader, model, epoch, save_path)
    #     # scheduler_focal.step()

