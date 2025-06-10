# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:41:48 2020

@author: Dian
"""
import time

import numpy as np

import torch
from scipy.io import loadmat

import os

from torch import nn

from calculate_metrics import Loss_SAM, Loss_RMSE, Loss_PSNR
# from mamba.othermodel.MIMO import Net

from deep_select_mamba import SRLF_Net

from utils import create_F, Gaussian_downsample, fspecial, AverageMeter


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



dataset = 'CAVE'
path="/media/hnu/code/data/CAVE/test/"


imglist = os.listdir(path)

model_path = r''    # saved model path
R = create_F()
R_inv = np.linalg.pinv(R)
R_inv = torch.Tensor(R_inv)
R = torch.Tensor(R)

net2=SRLF_Net(31,torch.Tensor(create_F()).cuda(), torch.Tensor(fspecial('gaussian', 7, 3)).cuda(),8).cuda()


checkpoint = torch.load(model_path)
net2.load_state_dict(checkpoint)
save_path = r"  "    # reconstructed image

RMSE = []
training_size = 256
stride = 256
PSF = fspecial('gaussian', 7, 3)
downsample_factor = 8

loss_func = nn.L1Loss(reduction='mean').cuda()


def reconstruction(net2, R, HSI_LR, MSI, HRHSI, downsample_factor, training_size, stride):
    index_matrix = torch.zeros((R.shape[1], MSI.shape[2], MSI.shape[3])).cuda()
    abundance_t = torch.zeros((R.shape[1], MSI.shape[2], MSI.shape[3])).cuda()
    a = []
    for j in range(0, MSI.shape[2] - training_size + 1, stride):
        a.append(j)
    a.append(MSI.shape[2] - training_size)
    b = []
    for j in range(0, MSI.shape[3] - training_size + 1, stride):
        b.append(j)
    b.append(MSI.shape[3] - training_size)
    for j in a:
        for k in b:
            temp_hrms = MSI[:, :, j:j + training_size, k:k + training_size]
            temp_lrhs = HSI_LR[:, :, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                        int(k / downsample_factor):int((k + training_size) / downsample_factor)]
            temp_hrhs = HRHSI[:, :, j:j + training_size, k:k + training_size]
            with torch.no_grad():
                # out,_,_= net2(temp_lrhs,temp_hrms)
                # out,_,_,_,_,_= net2(temp_lrhs,temp_hrms)

                out= net2(temp_lrhs,temp_hrms)
                # out= net2(temp_lrhs,temp_hrms)



                assert torch.isnan(out).sum() == 0

                HSI = out.squeeze()
                # 去掉维数为一的维度
                HSI = torch.clamp(HSI, 0, 1)
                abundance_t[:, j:j + training_size, k:k + training_size] = abundance_t[:, j:j + training_size,
                                                                           k:k + training_size] + HSI
                index_matrix[:, j:j + training_size, k:k + training_size] = 1 + index_matrix[:, j:j + training_size,
                                                                                k:k + training_size]

    HSI_recon = abundance_t / index_matrix
    assert torch.isnan(HSI_recon).sum() == 0
    return HSI_recon


val_loss = AverageMeter()
SAM = Loss_SAM()
RMSE = Loss_RMSE()
PSNR = Loss_PSNR()
sam = AverageMeter()
rmse = AverageMeter()
psnr = AverageMeter()




for i in range(0, len(imglist)):
    net2.eval()
    img = loadmat(path + imglist[i])
    if dataset=='CAVE':
        img1 = img["b"]
        # img1=img1/img1.max()
    elif dataset=='Harvard':
        img1 = img["ref"]
        img1=img1/img1.max()

    # print("real_hyper's shape =",img1.shape)

    HRHSI = torch.Tensor(np.transpose(img1, (2, 0, 1)))
    MSI = torch.tensordot(torch.Tensor(R), HRHSI, dims=([1], [0]))
    HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
    MSI_1 = torch.unsqueeze(MSI, 0)
    HSI_LR1 = torch.unsqueeze(torch.Tensor(HSI_LR), 0)  # 加维度 (b,c,h,w)
    time1 = time.time()
    to_fet_loss_hr_hsi = torch.unsqueeze(torch.Tensor(HRHSI), 0)

    with torch.no_grad():
        # with torch.profiler.profile(
        #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #         profile_memory=True
        # ) as prof:
            time1 = time.time()
            torch.cuda.synchronize()  # 确保 GPU 计算完成
            # prediction = reconstruction(net2, R, HSI_LR1.cuda(), MSI_1.cuda(), to_fet_loss_hr_hsi,
            #                                       downsample_factor, training_size, stride)
            prediction=net2(HSI_LR1.cuda(),MSI_1.cuda())
            torch.cuda.synchronize()  # 确保 GPU 计算完成
            time2 = time.time()
            # print(time2-time1)

            prediction=torch.clamp(prediction, 0, 1)
            Fuse = prediction.cpu().detach().numpy()
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # flops, params = profile(net2, inputs=(HSI_LR1.cuda(), MSI_1.cuda()))
    # print(flops, params)
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops, params)
    sam.update(SAM(np.transpose(HRHSI.cpu().detach().numpy(), (1, 2, 0)),
                   np.transpose(prediction.squeeze().cpu().detach().numpy(), (1, 2, 0))))
    rmse.update(RMSE(HRHSI.cpu().permute(1, 2, 0), prediction.squeeze().cpu().permute(1, 2, 0)))
    psnr.update(PSNR(HRHSI.cpu().permute(1, 2, 0), prediction.squeeze().cpu().permute(1, 2, 0)))

    faker_hyper = np.transpose(Fuse.squeeze(), (1, 2, 0))
    print(i, ':', imglist[i],faker_hyper.shape)
    print(PSNR(HRHSI.cpu().permute(1, 2, 0), prediction.squeeze().cpu().permute(1, 2, 0)))

    # test_data_path = os.path.join(save_path + imglist[i])
    # hdf5storage.savemat(test_data_path, {'fak': faker_hyper}, format='7.3')
    # hdf5storage.savemat(test_data_path, {'rea': img1}, format='7.3')
    # test_data_path = os.path.join(save_path + imglist[i]+'2')
    # hdf5storage.savemat(test_data_path, {'fak': faker_hyper}, format='7.3')
    # hdf5storage.savemat(test_data_path, {'rea': img1}, format='7.3')

print("val loss:",val_loss.avg)
print("val  PSNR:", psnr.avg.cpu().detach().numpy(), "  RMSE:", rmse.avg.cpu().detach().numpy(), "  SAM:", sam.avg)
print(torch.__version__)