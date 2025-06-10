

from scipy.io import loadmat
from thop import profile, clever_format

from calculate_metrics import Loss_SAM, Loss_RMSE, Loss_PSNR
from deep_select_mamba import SRLF_Net

from train_dataloader import *
from torch import nn
from tqdm import tqdm
import time
import pandas as pd
import torch.utils.data as data
from utils import create_F, fspecial
import math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def mkdir(path):

    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("训练文件夹为：{}".format(path))
    else:
        print('已存在{}'.format(path))


class MyarcLoss(torch.nn.Module):
    def __init__(self,R,PSF):
        super(MyarcLoss, self).__init__()
        self.R=R
        self.PSF=PSF
    def forward(self, output, target,MSI,sf):
        coeff = torch.unsqueeze(self.PSF, 0)
        coeff = torch.unsqueeze(coeff, 0)
        coeff = torch.repeat_interleave(coeff, output.shape[1],0)
        _,c, h, w= output.shape
        w1,h1=self.PSF.shape
        outs = nn.functional.conv2d(output, coeff.cuda(), bias=None, stride=sf, padding=int((w1-1)/2),  groups=c)
        target_HSI = nn.functional.conv2d(target, coeff.cuda(), bias=None, stride=sf, padding=int((w1-1)/2),  groups=c)
        RTZ = torch.tensordot(output, self.R, dims=([1], [1]))
        RTZ=torch.Tensor.permute(RTZ,(0,3,1,2))
        MSILoss=torch.mean(torch.abs(RTZ-MSI))
        tragetloss=torch.mean(torch.abs(output-target))
        HSILoss=torch.mean(torch.abs(outs[:,:,1:-1,1:-1]-target_HSI[:,:,1:-1,1:-1]))
        loss_total=MSILoss+tragetloss+0.1*HSILoss
        return loss_total


if __name__ == '__main__':
    # 路径参数
    root=os.getcwd()+"/train_save"
    model_name='deep_select_mamba'
    mkdir(os.path.join(root,model_name))
    current_list=os.listdir(os.path.join(root,model_name))
    for i in current_list:
        if len(i)>1:
            current_list.remove(i)

    current_list.append('0')
    int_list = [int(x) for x in current_list]
    train_value = max(int_list)+1
    model_name=os.path.join(model_name,str(train_value))


    # **************************************data_path ********************************


    # data_name='harvard'
    # path1 = 'D:\data\Harvard\harvard_train/'
    # path2 = 'D:\data\Harvard\harvard_test/'

    data_name ='cave'
    path1 = '/media/hnu/code/data/CAVE/train/'
    path2 = '/media/hnu/code/data/CAVE/test/'

    # ************************************** training set *******************************
    loss_func2 = nn.L1Loss(reduction='mean').cuda()
    R = create_F()
    PSF = fspecial('gaussian', 7, 3)

    loss_func=MyarcLoss(torch.Tensor(R).cuda(),torch.Tensor(PSF).cuda()).cuda()

    downsample_factor = 8
    training_size = 64
    stride = 32
    stride1 = 32
    LR = 4e-4
    EPOCH = 2000
    weight_decay=1e-8
    BATCH_SIZE = 16
    num = 20
    psnr_optimal = 47
    rmse_optimal = 1.5

    test_epoch=300
    val_interval = 50
    checkpoint_interval = 100




    path=os.path.join(root,model_name)
    mkdir(path)
    pkl_name=data_name+'_pkl'
    pkl_path=os.path.join(path,pkl_name)
    os.makedirs(pkl_path)
    # ******************88888record in excel**************************88
    df = pd.DataFrame(columns=['epoch', 'lr', 'train_loss','val_loss','val_rmse', 'val_psnr', 'val_sam'])  # 列名
    excel_name=data_name+'_record.csv'
    excel_path=os.path.join(path,excel_name)
    df.to_csv(excel_path, index=False)

    df2 = pd.DataFrame(columns=['epoch', 'lr', 'train_loss'])  # 列名
    excel_name2=data_name+'_train_record.csv'
    excel_path2=os.path.join(path,excel_name2)
    df2.to_csv(excel_path2, index=False)


    cnn=SRLF_Net(31,torch.Tensor(R).cuda(),torch.Tensor(PSF).cuda(),8).cuda()

    hsi = torch.randn((2, 31, 8, 8), device='cuda')
    pan = torch.randn((2, 3, 64, 64), device='cuda')
    target=torch.randn((2, 31,64,64), device='cuda')

    # model = Feafusion(31, 8).cuda()
    model=SRLF_Net(31,torch.Tensor(R).cuda(),torch.Tensor(PSF).cuda(),8).cuda()

    flops, params = profile(model, inputs=(hsi, pan))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
    del model,hsi,pan,target

    train_data = CAVEHSIDATAprocess(path1, R, training_size, stride, downsample_factor, PSF, num)
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    maxiteration = math.ceil(len(train_data) / BATCH_SIZE) * EPOCH
    print("maxiteration：", maxiteration)
    # Model initialization
    for m in cnn.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR,betas=(0.9, 0.999),weight_decay=weight_decay)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, maxiteration, eta_min=1e-6, last_epoch=-1)







    for epoch in range(1, EPOCH+1):
        cnn.train()
        loss_all = []
        loop = tqdm(train_loader, total=len(train_loader),ncols=120)
        for train_hrhs, train_hrms, train_lrhs in loop:
            a1=train_hrhs
            a2=train_hrms
            a3=train_lrhs

            lr = optimizer.param_groups[0]['lr']
            output = cnn(a3.cuda(),a2.cuda())
            loss = loss_func2(output, a1.cuda())

            loss_temp = loss
            loss_all.append(np.array(loss_temp.detach().cpu().numpy()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch [{epoch}/{EPOCH}]')
            loop.set_postfix({'loss': '{0:1.8f}'.format(np.mean(loss_all)), "lr": '{0:1.8f}'.format(lr)})
            scheduler.step()
        train_list = [epoch, lr,np.mean(loss_all)]
        train_record = pd.DataFrame([train_list])
        train_record.to_csv(excel_path2,mode='a', header=False, index=False)


        if ((epoch % val_interval == 0) and (epoch>=test_epoch) ) or epoch==1:
            cnn.eval()
            val_loss=AverageMeter()
            SAM = Loss_SAM()
            RMSE = Loss_RMSE()
            PSNR = Loss_PSNR()
            sam = AverageMeter()
            rmse = AverageMeter()
            psnr = AverageMeter()

            imglist = os.listdir(path2)
            with torch.no_grad():
                for i in range(0, len(imglist)):
                    img = loadmat(path2 + imglist[i])
                    img1 = img["b"]
                    img1 = img1 / img1.max()
                    HRHSI = torch.Tensor(np.transpose(img1, (2, 0, 1)))
                    MSI = torch.tensordot(torch.Tensor(R), HRHSI, dims=([1], [0]))
                    HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
                    MSI_1 = torch.unsqueeze(MSI, 0)
                    HSI_LR1 = torch.unsqueeze(torch.Tensor(HSI_LR), 0)
                    to_fet_loss_hr_hsi=torch.unsqueeze(torch.Tensor(HRHSI), 0)

                    prediction,val_loss = reconstruction(cnn, torch.Tensor(R),torch.Tensor(PSF), HSI_LR1.cuda(), MSI_1.cuda(), to_fet_loss_hr_hsi,downsample_factor, training_size, stride1,val_loss)
                    sam.update(SAM(np.transpose(HRHSI.cpu().detach().numpy(),(1, 2, 0)),np.transpose(prediction.squeeze().cpu().detach().numpy(),(1, 2, 0))))
                    rmse.update(RMSE(HRHSI.cpu().permute(1,2,0),prediction.squeeze().cpu().permute(1,2,0)))
                    psnr.update(PSNR(HRHSI.cpu().permute(1,2,0),prediction.squeeze().cpu().permute(1,2,0)))

                if  epoch == 1:
                    torch.save(cnn.state_dict(),pkl_path +'/'+ str(epoch) + 'EPOCH' + '_PSNR_best.pkl')

                if torch.abs(psnr_optimal-psnr.avg)<0.15:
                    torch.save(cnn.state_dict(), pkl_path + '/' + str(epoch) + 'EPOCH' + '_PSNR_best.pkl')
                if psnr.avg > psnr_optimal:
                    psnr_optimal = psnr.avg

                if torch.abs(rmse.avg-rmse_optimal)<0.15:
                    torch.save(cnn.state_dict(),pkl_path +'/'+ str(epoch) + 'EPOCH' + '_RMSE_best.pkl')
                if rmse.avg < rmse_optimal:
                    rmse_optimal = rmse.avg



                print("val  PSNR:",psnr.avg.cpu().detach().numpy(), "  RMSE:", rmse.avg.cpu().detach().numpy(), "  SAM:", sam.avg,"val loss:", val_loss.avg.cpu().detach().numpy())
                val_list = [epoch, lr,np.mean(loss_all),val_loss.avg.cpu().detach().numpy(),rmse.avg.cpu().detach().numpy(), psnr.avg.cpu().detach().numpy(), sam.avg]

                val_data = pd.DataFrame([val_list])
                val_data.to_csv(excel_path,mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
                time.sleep(0.1)


