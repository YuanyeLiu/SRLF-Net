import os
import numpy as np
import torch
from hdf5storage import loadmat
from torch import nn

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        Itrue = im_true.clamp(0., 1.)*data_range
        Ifake = im_fake.clamp(0., 1.)*data_range
        err=Itrue-Ifake
        err=torch.pow(err,2)
        err = torch.mean(err,dim=0)
        err = torch.mean(err,dim=0)

        psnr = 10. * torch.log10((data_range ** 2) / err)
        psnr=torch.mean(psnr)
        return psnr


class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs.clamp(0., 1.)*255- label.clamp(0., 1.)*255
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
        return rmse

class Loss_SAM(nn.Module):
    def __init__(self):
        super(Loss_SAM, self).__init__()
        self.eps=2.2204e-16
    def forward(self,im1, im2):
        assert im1.shape == im2.shape
        H,W,C=im1.shape
        im1 = np.reshape(im1,(H*W,C))
        im2 = np.reshape(im2,(H*W,C))
        core=np.multiply(im1, im2)
        mole = np.sum(core, axis=1)
        im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
        im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
        deno = np.multiply(im1_norm, im2_norm)
        sam = np.rad2deg(np.arccos(((mole+self.eps)/(deno+self.eps)).clip(-1,1)))
        return np.mean(sam)


class Loss_SAM2(nn.Module):
    def __init__(self):
        super(Loss_SAM2, self).__init__()
        self.eps=2.2204e-16
    def forward(self,im1, im2):
        assert im1.shape == im2.shape
        H,W,C=im1.shape
        im1 = np.reshape(im1,(H*W,C)).transpose(1,0)
        im2 = np.reshape(im2,(H*W,C)).transpose(1,0)
        core=np.multiply(im1, im2)
        mole = np.sum(core, axis=1)
        im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
        im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
        deno = np.multiply(im1_norm, im2_norm)
        sam = np.rad2deg(np.arccos(((mole+self.eps)/(deno+self.eps)).clip(-1,1)))
        return np.mean(sam)


class Loss_SAM3(nn.Module):
    def __init__(self):
        super(Loss_SAM3, self).__init__()
        self.eps = 2.2204e-16#torch.finfo(torch.float32).eps  # Minimum positive value of torch float32
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=self.eps)

    def forward(self, im1, im2):
        assert im1.shape == im2.shape
        B, C, H, W = im1.shape

        # Reshape images
        im1 = im1.contiguous().view(B, C, -1) # Shape: [B, H*W, C]
        im2 = im2.contiguous().view(B, C, -1) # Shape: [B, H*W, C]

        # Compute cosine similarity
        core = torch.mul(im1, im2)  # Element-wise multiplication
        mole = torch.sum(core, dim=2)  # Sum along the channel dimension
        # im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
        # im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
        im1_norm = torch.sqrt(torch.sum(torch.square(im1), dim=2))
        im2_norm = torch.sqrt(torch.sum(torch.square(im2), dim=2))
        deno = torch.mul(im1_norm, im2_norm)

        # Compute SAM
        sam = torch.acos(((mole + self.eps) / (deno + self.eps)).clamp(-1, 1))
        sam_deg = torch.rad2deg(sam)

        # Compute mean over the batch and return
        return sam_deg
