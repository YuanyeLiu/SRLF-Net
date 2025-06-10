
from thop import profile, clever_format

from scipy.io import loadmat

from calculate_metrics import Loss_SAM, Loss_RMSE, Loss_PSNR


import hdf5storage as h5

from deep_select_mamba_gf5 import SRLF_Net
from train_dataloader import *
from torch import nn
from tqdm import tqdm
import time
import pandas as pd
import torch.utils.data as data
from utils import create_F, fspecial
import math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = "0" 


def mkdir(path):

    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("训练文件夹为：{}".format(path))
    else:
        print('已存在{}'.format(path))



if __name__ == '__main__':
    # 路径参数
    root=os.getcwd()+"/gf5_zs_train_save"
    model_name='deep_select_mamba_gf5'
    mkdir(os.path.join(root,model_name))
    ori_list=os.listdir(os.path.join(root,model_name))
    current_list=[]
    for i in ori_list:
        if len(i)<=2:
            current_list.append(i)

    del ori_list



    current_list.append('0')
    int_list = [int(x) for x in current_list]
    train_value = max(int_list)+1
    model_name=os.path.join(model_name,str(train_value))

    # data_name='harvard'
    # path1 = 'D:\data\Harvard\harvard_train/'
    # path2 = 'D:\data\Harvard\harvard_test/'


    training_size=64
    stride=32
    downsample_factor=2
    data_name='gf5'
    HSI = np.load(r"/media/hnu/code/data/GF5-GF1-new/GF5-GF1/reg_msi.npy")
    MSI = np.load(r"/media/hnu/code/data/GF5-GF1-new/GF5-GF1/reg_pan.npy")

    R = np.load(r"/media/hnu/code/data/GF5-GF1-new/GF5-GF1/R.npy")
    C = np.load(r"/media/hnu/code/data/GF5-GF1-new/GF5-GF1/C.npy")
    R = np.transpose(R, (1, 0))
    PSF=C
    HSI_LR = Gaussian_downsample(np.transpose(HSI, (2, 0, 1)),  C, downsample_factor)
    LRMSI = Gaussian_downsample(np.transpose(MSI, (2, 0, 1)),  C, downsample_factor)

    test_HRHSI0=np.transpose(HSI,(2, 0, 1))
    test_HRMSI0=LRMSI
    test_LRHSI0=HSI_LR


    print("训练数据处理完成")


    # 训练参数
    loss_func = nn.L1Loss(reduction='mean').cuda()



    stride1 = 32
    LR = 0.00015877942650285176
    EPOCH = 2000
    weight_decay=1e-8    # 我的模型是1e-8
    BATCH_SIZE =8
    psnr_optimal = 40
    rmse_optimal = 4.5

    test_epoch=100
    val_interval = 100           # 每隔val_interval epoch测试一次
    checkpoint_interval = 100
    # maxiteration = math.ceil(((512 - training_size) // stride + 1) ** 2 * num / BATCH_SIZE) * EPOCH


    # warm_lr_scheduler
    decay_power = 1.5
    init_lr2 = 5e-4
    init_lr1 = 5e-5
    min_lr=0

    hsi = torch.randn((2, 150, 32, 32), device='cuda')
    pan = torch.randn((2, 4, 64, 64), device='cuda')

    model=SRLF_Net(150,torch.Tensor(R).cuda(),torch.Tensor(PSF).cuda(),2).cuda()

    flops, params = profile(model, inputs=(hsi, pan))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
    del model,hsi,pan


    path=os.path.join(root,model_name)
    mkdir(path)
    pkl_name=data_name+'_pkl'
    pkl_path=os.path.join(path,pkl_name)
    os.makedirs(pkl_path)


    df = pd.DataFrame(columns=['epoch', 'lr', 'train_loss','val_loss','val_rmse', 'val_psnr', 'val_sam'])  # 列名
    excel_name=data_name+'_record.csv'
    excel_path=os.path.join(path,excel_name)
    df.to_csv(excel_path, index=False)

    df2 = pd.DataFrame(columns=['epoch', 'lr', 'train_loss'])  # 列名
    excel_name2=data_name+'_train_record.csv'
    excel_path2=os.path.join(path,excel_name2)
    df2.to_csv(excel_path2, index=False)

    train_data=RealDATAProcess2(HSI,MSI,training_size, stride, downsample_factor,C)
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    maxiteration = math.ceil(len(train_data)/BATCH_SIZE)* EPOCH
    print("maxiteration：", maxiteration)
    warm_iter = math.floor(maxiteration / 40)



    cnn=SRLF_Net(150,torch.Tensor(R).cuda(),torch.Tensor(PSF).cuda(),2).cuda()

    # 模型初始化
    for m in cnn.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR,betas=(0.9, 0.999),weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, maxiteration, eta_min=0, last_epoch=-1)








    for epoch in range(1, EPOCH+1):
        cnn.train()
        loss_all = []
        loop = tqdm(train_loader, total=len(train_loader))
        for a1, a2, a3 in loop:

            lr = optimizer.param_groups[0]['lr']

            output = cnn(a3.cuda(), a2.cuda())   # hsrnet
            loss = loss_func(output, a1.cuda())

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
        train_record.to_csv(excel_path2,mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了



        if ((epoch % val_interval == 0) and (epoch>=test_epoch) ) or epoch==1:
            cnn.eval()
            val_loss=AverageMeter()
            SAM = Loss_SAM()
            RMSE = Loss_RMSE()
            PSNR = Loss_PSNR()
            sam = AverageMeter()
            rmse = AverageMeter()
            psnr = AverageMeter()

            with torch.no_grad():
                # img1 = img1 / img1.max()
                test_HRHSI = torch.unsqueeze(torch.Tensor(test_HRHSI0),0)
                test_HRMSI =torch.unsqueeze(torch.Tensor(test_HRMSI0),0)
                test_LRHSI=torch.unsqueeze(torch.Tensor(test_LRHSI0),0)


                prediction,val_loss = reconstruction_fg5(cnn, R, test_LRHSI.cuda(), test_HRMSI.cuda(), test_HRHSI,downsample_factor, training_size, stride1,val_loss)
                # print(Fuse.shape)
                prediction=torch.round(prediction*255)/255.0

                sam.update(SAM(np.transpose(test_HRHSI.squeeze().cpu().detach().numpy(),(1, 2, 0)),np.transpose(prediction.squeeze().cpu().detach().numpy(),(1, 2, 0))))
                rmse.update(RMSE(test_HRHSI.squeeze().cpu().permute(1,2,0),prediction.squeeze().cpu().permute(1,2,0)))
                psnr.update(PSNR(test_HRHSI.squeeze().cpu().permute(1,2,0),prediction.squeeze().cpu().permute(1,2,0)))

            if  epoch == 1 or epoch==EPOCH:
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
