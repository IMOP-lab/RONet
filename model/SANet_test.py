import torch
import torch.nn.functional as F
import torch.nn as nn
# import sys
# sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.RONet import RONet
from data import Test_dataset, Test_dataset2
import torch.utils.data as data


parser = argparse.ArgumentParser()
# parser.add_argument('--testsize', type=int, default=352, help='testing size')
# parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='/home/wanbin/wanbin/new_data/rgbt/VT5000/VT5000_clear/Test/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path



# load the model
model = nn.DataParallel(RONet(3))
image_root = dataset_path + '/RGB/'
gt_root = dataset_path +'/GT/'
depth_root=dataset_path +'/T/'
test_dataset=Test_dataset(image_root, gt_root,depth_root, 384)
batch= 48
data_loader= data.DataLoader(dataset=test_dataset,
    batch_size=batch,
    shuffle=False,
    num_workers=4,
    pin_memory=True)
for checkp in range(62, 120+1):
    if checkp % 2 ==0:
    # if checkp in [12,18,96,102,108,150,204]:
        with torch.no_grad():
            modelpath= './BBSNet_cpts/BBSNet_epoch_' + str(checkp)+ '.pth'
            model.load_state_dict(torch.load(modelpath))
            # model_fuse.load_state_dict(torch.load('/home/lvchengtao/code/BBS-Net-master-2/BBSNet_cpts/BBSNet_epoch_36_3.pth'))
            # model.cuda()
            model.cuda()
            # model.eval()
            model.eval()
            save_path = './test_maps/VT5000/'+str(checkp)+ '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            #test
            # test_datasets = ['NJU2K','NLPR','STERE', 'DES', 'SSD','LFSD','SIP']
            # for dataset in test_datasets:
            loss_e= 0

            for i, (image, gt_shape, depth, name) in enumerate(data_loader):
                print('batch {:4d}'.format(i))
                image = image.cuda()
                depth = depth.cuda()
                # print(gt_shape)
                # print((gt_shape[1][0].item(),gt_shape[2][0].item()))
                bz= image.shape[0]
                #print(bz)
                pred,_,_,_,_ = model(image, depth)
                #print(pred.size())

                # pred= F.interpolate(pred,size=224,mode='bilinear')
                for j in range(bz):
                    
                    res2= pred[j]
                    # res2= F.interpolate(res2.unsqueeze(dim=0),size=(gt_shape[1][j].item(),gt_shape[2][j].item()),mode='bilinear')
                    res2 = torch.sigmoid(res2).data.cpu().numpy().squeeze()
                    name2= name[j]
                    # print(name2)
                    res2 = (res2 - res2.min()) / (res2.max() - res2.min() + 1e-8)
                    res2= res2*255
                    cv2.imwrite(save_path+ name2, res2)
                # break
            # print('Loss_e: {:.4f}, '.format(loss_e))
            print('Test Done!', checkp)


#####VT1000
dataset_path= '/home/wanbin/wanbin/new_data/rgbt/VT1000/'
image_root = dataset_path + '/RGB/'
gt_root = dataset_path +  '/GT/'
depth_root=dataset_path +'/T/'
test_dataset=Test_dataset2(image_root, gt_root,depth_root, 384)
batch= 48
data_loader= data.DataLoader(dataset=test_dataset,
    batch_size=batch,
    shuffle=False,
    num_workers=4,
    pin_memory=True)
for checkp in range(62, 120+1):
    if checkp % 2 ==0:
    # if checkp in [12,18,96,102,108,150,204]:
        with torch.no_grad():
            modelpath= './BBSNet_cpts/BBSNet_epoch_' + str(checkp)+ '.pth'
            model.load_state_dict(torch.load(modelpath))
            # model_fuse.load_state_dict(torch.load('/home/lvchengtao/code/BBS-Net-master-2/BBSNet_cpts/BBSNet_epoch_36_3.pth'))
            # model.cuda()
            model.cuda()
            # model.eval()
            model.eval()
            save_path = './test_maps/VT1000/'+str(checkp)+ '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            loss_e= 0

            for i, (image, gt_shape, depth, name) in enumerate(data_loader):
                # print(gt_shape)
                print('batch {:4d}'.format(i))
                image = image.cuda()
                depth = depth.cuda()
                # print(gt_shape)
                # print((gt_shape[1][0].item(),gt_shape[2][0].item()))
                bz= image.shape[0]
                pred,_,_,_,_ = model(image, depth)
               
                # pred= F.interpolate(pred,size=224,mode='bilinear')
                for j in range(bz):
                    res2= pred[j]
                    # res2= F.interpolate(res2.unsqueeze(dim=0),size=(gt_shape[1][j].item(),gt_shape[2][j].item()),mode='bilinear')
                    res2 = torch.sigmoid(res2).data.cpu().numpy().squeeze()
                    name2= name[j]
                    # print(name2)
                    res2 = (res2 - res2.min()) / (res2.max() - res2.min() + 1e-8)
                    res2= res2*255
                    cv2.imwrite(save_path+ name2, res2)
                # break
            # print('Loss_e: {:.4f}, '.format(loss_e))
            print('Test Done!', checkp)



# # #####VT821
dataset_path= '//home/wanbin/wanbin/new_data/rgbt/VT821/'
image_root = dataset_path + '/RGB/'
gt_root = dataset_path +  '/GT/'
depth_root=dataset_path +'/T/'
test_dataset=Test_dataset2(image_root, gt_root,depth_root, 384)
batch= 48
data_loader= data.DataLoader(dataset=test_dataset,
    batch_size=batch,
    shuffle=False,
    num_workers=4,
    pin_memory=True)
for checkp in range(62, 120+1):
    if checkp % 2 ==0:
    # if checkp in [12,18,96,102,108,150,204]:
        with torch.no_grad():
            modelpath= './BBSNet_cpts/BBSNet_epoch_' + str(checkp)+ '.pth'
            model.load_state_dict(torch.load(modelpath))
            # model_fuse.load_state_dict(torch.load('/home/lvchengtao/code/BBS-Net-master-2/BBSNet_cpts/BBSNet_epoch_36_3.pth'))
            # model.cuda()
            model.cuda()
            # model.eval()
            model.eval()
            save_path = './test_maps/VT821/'+str(checkp)+ '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            loss_e= 0

            for i, (image, gt_shape, depth, name) in enumerate(data_loader):
                print('batch {:4d}'.format(i))
                image = image.cuda()
                depth = depth.cuda()
                # print(gt_shape)
                # print((gt_shape[1][0].item(),gt_shape[2][0].item()))
                bz= image.shape[0]
                pred,_,_,_,_ = model(image, depth)
           
                # pred= F.interpolate(pred,size=224,mode='bilinear')
                for j in range(bz):
                    res2= pred[j]
                    # res2= F.interpolate(res2.unsqueeze(dim=0),size=(gt_shape[1][j].item(),gt_shape[2][j].item()),mode='bilinear')
                    res2 = torch.sigmoid(res2).data.cpu().numpy().squeeze()
                    name2= name[j]
                    # print(name2)
                    res2 = (res2 - res2.min()) / (res2.max() - res2.min() + 1e-8)
                    res2= res2*255
                    cv2.imwrite(save_path+ name2, res2)
                # break
            # print('Loss_e: {:.4f}, '.format(loss_e))
            print('Test Done!', checkp)