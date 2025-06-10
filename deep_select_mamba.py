import time

import numpy as np
import torch
from einops import rearrange
from thop import profile, clever_format
from torch import nn

from utils import Gaussian_downsample, fspecial, create_F
from mambair import SS2D
import torch.nn.functional as F
from ssim import SSIM
import matplotlib.pyplot as plt

class Degenerate(torch.nn.Module):
    def __init__(self):
        super(Degenerate, self).__init__()
        self.c=31
    def forward(self, output, R,coeff,sf,w1):
        PLR = nn.functional.conv2d(output, coeff, bias=None, stride=sf, padding=int((w1-1)/2),  groups=self.c)
        PMSI = torch.tensordot(output, R, dims=([1], [1]))
        PMSI=torch.Tensor.permute(PMSI,(0,3,1,2))
        return PLR,PMSI


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B*N, C)

    x[idx1.reshape(-1)] = x1.reshape(B*N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B*N2, C)

    x = x.reshape(B, N, C)
    return x


def batch_index_fill2(x, x1, idx1,):
    B, N, C = x.size()
    B, N1, C = x1.size()


    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N


    x = x.reshape(B*N, C)

    x[idx1.reshape(-1)] = x1.reshape(B*N1, C)
    x = x.reshape(B, N, C)
    return x


class Loss_SAM3(nn.Module):
    def __init__(self):
        super(Loss_SAM3, self).__init__()
        self.eps = 2.2204e-16#torch.finfo(torch.float32).eps  # Minimum positive value of torch float32
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=self.eps)

    def forward(self, im1, im2):
        assert im1.shape == im2.shape
        B, C, H, W = im1.shape

        # Reshape images
        im1 = im1.contiguous().view(B, C, -1).permute(0,2,1) # Shape: [B, H*W, C]
        im2 = im2.contiguous().view(B, C, -1).permute(0,2,1) # Shape: [B, H*W, C]

        core = torch.mul(im1, im2)  # Element-wise multiplication
        mole = torch.sum(core, dim=2)  # Sum along the channel dimension

        im1_norm = torch.sqrt(torch.sum(torch.square(im1), dim=2))
        im2_norm = torch.sqrt(torch.sum(torch.square(im2), dim=2))
        deno = torch.mul(im1_norm, im2_norm)

        sam = torch.acos(((mole + self.eps) / (deno + self.eps)).clamp(-1, 1))
        sam_deg = torch.rad2deg(sam)

        return sam_deg

class MSA(nn.Module):
    def __init__(self, num_vector, num_heads_column, heads_number):

        super(MSA, self).__init__()
        self.num_vector = num_vector
        self.num_heads_column = num_heads_column
        self.heads_number = heads_number
        self.to_q = nn.Linear(num_vector, num_heads_column * heads_number, bias=False)
        self.to_k = nn.Linear(num_vector, num_heads_column * heads_number, bias=False)
        self.to_v = nn.Linear(num_vector, num_heads_column * heads_number, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads_number, 1, 1))  # 权重参数*CORE
        self.proj = nn.Linear(num_heads_column * heads_number, num_vector)
        self.pos_emb = nn.Sequential(
            nn.Linear(num_heads_column * heads_number, num_vector),
            nn.modules.activation.GELU(),
            nn.Linear(num_vector, num_vector),
        )

    def forward(self, x_in):

        b, n, c = x_in.shape
        x = x_in

        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads_number),
                      (q_inp, k_inp, v_inp))
        v = v
        q = q.transpose(-2, -1)  # q,k,v: b,heads,c,hw
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, n, self.heads_number * self.num_heads_column)
        out_c = self.proj(x)

        out_p = self.pos_emb(v_inp)

        return out_c+out_p

class Transformer(nn.Module):
    def __init__(self,x_channel):
        super(Transformer,self).__init__()
        self.saln1 = nn.LayerNorm(x_channel)
        self.saln2 = nn.LayerNorm(x_channel)
        self.sa=MSA(x_channel,x_channel//4,4)
        self.re_conv1=nn.Sequential(
            nn.Linear(x_channel,x_channel//2,bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(x_channel//2,x_channel,bias=False),
        )

    def forward(self,v1):
        nor_v1=self.saln1(v1)
        re_fea1=self.sa(nor_v1)+v1
        norre_fea1=self.saln2(re_fea1)
        refine1=self.re_conv1(norre_fea1)+re_fea1

        return refine1

class Mamba(nn.Module):
    def __init__(self, x_channel):
        super(Mamba, self).__init__()
        self.ssm1 = SS2D(d_model=x_channel, d_state=x_channel // 2, expand=2, dropout=0.3)
        self.preln1 = nn.LayerNorm(x_channel)

    def forward(self,fufea1):
        nor_fufea1 = self.preln1(fufea1.permute(0, 2, 3, 1))
        fea1 = self.ssm1(nor_fufea1).permute(0, 3, 1, 2) + fufea1
        return fea1







class Model_and_SSIM_Guided(nn.Module):
    def __init__(self,x_channel):
        super(Model_and_SSIM_Guided,self).__init__()

        self.mamba_64=Mamba(x_channel)
        self.down = nn.Sequential(
            # 4倍
            nn.Conv2d(x_channel, x_channel, kernel_size=6, stride=4, padding=2, bias=False),
            # 2倍数
            nn.Conv2d(x_channel, x_channel, (4, 4), (2, 2), 1, bias=False)
        )
        self.mamba_8=Mamba(x_channel)


        self.up = nn.Sequential(
            nn.Conv2d(x_channel, x_channel*64, 1)
        )
        self.ps=nn.PixelShuffle(8)

        self.mambacat=nn.Conv2d(x_channel*2,x_channel,1)
        self.mambafusion=Mamba(x_channel)

        self.layernorm=nn.LayerNorm(x_channel)
        self.to_r_ssim=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.to_r_sam=nn.AdaptiveAvgPool2d((1,1))

        self.pancat=nn.Conv2d(x_channel+3,x_channel,1)

        self.ssim_refine=Transformer(x_channel)
        self.sam_refine=Transformer(x_channel)
        self.hsicat=nn.Conv2d(x_channel*2,x_channel,1)

        self.layernorm_score=nn.LayerNorm(1)

        self.catconv1=nn.Linear(x_channel*2,x_channel,bias=False)




        self.ssim=SSIM(5)
        self.sam=Loss_SAM3()
        self.to_hsi_and_msi=Degenerate()

        self.down2 = nn.Sequential(
            nn.Conv2d(x_channel, x_channel, kernel_size=6, stride=4, padding=2, bias=False),
            nn.Conv2d(x_channel, x_channel, (4, 4), (2, 2), 1, bias=False)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(x_channel, x_channel*64, 1)
        )
        self.ps2=nn.PixelShuffle(8)
        self.convout=nn.Sequential(
            nn.Conv2d(x_channel*2,x_channel,1)
        )
        self.mambafusion2=Mamba(x_channel)
        self.convout2=nn.Sequential(
            nn.Conv2d(x_channel*2,x_channel,1)
        )


    def forward(self,fufea_64,R,coeff,sf,w1,gt_msi,gt_hsi):
        global stage,stage1
        b,c,h,w=fufea_64.shape
        fufea64=self.mamba_64(fufea_64)
        fufea8=self.down(fufea64)
        fufea8=self.mamba_8(fufea8)
        upfufea8=self.up(fufea8)
        upfufea8=self.ps(upfufea8)

        pre_fusion=self.mambacat(torch.cat([fufea64,upfufea8],dim=1))
        pre_fusion=self.mambafusion(pre_fusion)

        pre_fusion=self.layernorm(pre_fusion.permute(0,2,3,1)).permute(0,3,1,2)
        _, PMSI = self.to_hsi_and_msi(pre_fusion,R,coeff,sf,w1)
        PMSI=PMSI.clamp(0,1)

        ssim_score=self.ssim(PMSI,gt_msi)

        num_keep_node_bili=0.3*(1-self.to_r_ssim(ssim_score).reshape(-1))

        ssim_score=ssim_score.reshape(b, -1)
        B1,N1=ssim_score.shape
        ssim_idx = torch.argsort(ssim_score, dim=1, descending=False)




        num_keep_node=int(torch.mean(N1*num_keep_node_bili))

        ssim_idx1 = ssim_idx[:, :num_keep_node]      # bad points


        fea_add_pan=self.pancat(torch.cat([pre_fusion,gt_msi],dim=1))


        ssim_v1= batch_index_select(fea_add_pan.reshape(b, c, -1).permute(0, 2, 1), ssim_idx1)     # buhaode



        ssim_refine=self.ssim_refine(ssim_v1)
        ssim_out = torch.zeros_like(fufea_64)
        ssim_out= batch_index_fill2(ssim_out.reshape(b, c, -1).permute(0, 2, 1), ssim_refine,  ssim_idx1)
        ssim_out=ssim_out.permute(0, 2, 1).reshape(b,c,h,w)+fea_add_pan

        PLR, _ = self.to_hsi_and_msi(ssim_out,R,coeff,sf,w1)

        sam_score=self.sam(PLR,gt_hsi).reshape(b,h//8,w//8,-1)

        sam_score=sam_score.reshape(b,-1)
        B2, N2 = sam_score.shape
        num_keep_node=int(torch.mean(N2*num_keep_node_bili))
        sam_idx = torch.argsort(sam_score, dim=1, descending=True)  #降排列
        sam_idx1 = sam_idx[:, :num_keep_node]    # 不好的


        samfea=self.hsicat(torch.cat([PLR,gt_hsi],dim=1))

        sam_v1 = batch_index_select(samfea.reshape(b, c, -1).permute(0, 2, 1), sam_idx1)  #

        sam_refine = self.sam_refine(sam_v1)

        sam_out=torch.zeros_like(fufea8)

        sam_out= batch_index_fill2(sam_out.reshape(b, c, -1).permute(0, 2, 1), sam_refine,  sam_idx1)


        sam_out=sam_out.permute(0, 2, 1).reshape(b,c,h//8,w//8)+samfea
        sam_out=self.up2(sam_out)
        sam_out=self.ps(sam_out)

        fu_all=torch.cat([ssim_out,sam_out],dim=1)
        fu_all=self.convout(fu_all)
        re_ssm=self.mambafusion2(fu_all)+fu_all
        out=self.convout2(torch.cat([re_ssm,fufea_64],dim=1))
        return out






class SRLF_Net(nn.Module):
    def __init__(self,x_channel,R,PSF,sf):
        super(SRLF_Net,self).__init__()
        self.sf=sf
        self.R=R
        self.PSF=PSF
        self.fu64=nn.Conv2d(x_channel+3,x_channel,3,1,1)

        self.icp1=Model_and_SSIM_Guided(x_channel)
        self.icp2=Model_and_SSIM_Guided(x_channel)
        self.icp3=Model_and_SSIM_Guided(x_channel)
        self.icp4=Model_and_SSIM_Guided(x_channel)
        self.refine=nn.Sequential(
            nn.Conv2d(x_channel*5,x_channel,1, 1, 0, bias=False)
        )


    def forward(self,hsi,pan):
        coeff = torch.unsqueeze(self.PSF, 0)
        coeff = torch.unsqueeze(coeff, 0)
        coeff = torch.repeat_interleave(coeff, hsi.shape[1], 0)
        _, _, h, w = pan.shape
        _, c, _,_ = hsi.shape
        w1, h1 = self.PSF.shape
        uphsi = torch.nn.functional.interpolate(hsi, scale_factor= self.sf, mode='bicubic')
        catfea64=torch.cat((uphsi,pan),dim=1)
        catfea64=self.fu64(catfea64)
        fu1=self.icp1(catfea64,self.R,coeff,self.sf,w1,pan,hsi)
        fu2=self.icp2(fu1,self.R,coeff,self.sf,w1,pan,hsi)
        fu3=self.icp3(fu2,self.R,coeff,self.sf,w1,pan,hsi)
        fu4=self.icp4(fu3,self.R,coeff,self.sf,w1,pan,hsi)
        refine=self.refine(torch.cat((fu4,fu3,fu2,fu1,catfea64),dim=1))
        out=refine + uphsi
        return out


if __name__=='__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")


    hsi = torch.randn((1, 31,512//8,512//8), device=device)
    pan = torch.randn((1,3,512,512), device=device)

    R=torch.Tensor(create_F()).cuda()
    PSF=torch.Tensor(fspecial('gaussian', 7, 3)).cuda()
    model=Feafusion(31,R,PSF,8).cuda()
    model.eval()

    with torch.no_grad():
        out=model(hsi,pan)

    with torch.no_grad():
        flops, params = profile(model, inputs=(hsi,pan))
        print(flops, params)
        flops, params = clever_format([flops, params], "%.3f")
        print(flops, params)


    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter: % .4fM' % (total / 1e6))

