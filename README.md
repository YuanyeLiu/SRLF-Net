# The code for 《A Selective Re-learning Mechanism for Hyperspectral Fusion Imaging》.

## Authors
Yuanye Liu, Jinyang Liu, Renwei Dian\*, Shutao Li

###  Quantitative indexes of the test methods on the CAVE and Harvard datasets

> The best result is marked in **bold**.

| Method       | FLOPS     | Params (M) | PSNR (CAVE) | SAM (CAVE) | UIQI (CAVE) | SSIM (CAVE) | PSNR (Harvard) | SAM (Harvard) | UIQI (Harvard) | SSIM (Harvard) |
|--------------|-----------|------------|-------------|------------|-------------|-------------|----------------|----------------|----------------|----------------|
| Hysure       | /         | /          | 42.9303     | 8.1108     | 0.9321      | 0.9814      | 45.2041        | 3.6180         | 0.8689         | 0.9807         |
| NSSR         | /         | /          | 46.6246     | 3.2505     | 0.9563      | 0.9910      | 47.1459        | 2.9396         | 0.8899         | 0.9843         |
| DAEM         | 417.5G\*  | **0.02**   | 41.5822     | 4.1965     | 0.9283      | 0.9796      | 43.3745        | 4.3586         | 0.8467         | 0.9728         |
| FusionMamba  | 129.62G   | 2.58       | 46.8107     | 2.6511     | 0.9710      | 0.9937      | 47.6266        | 2.7612         | 0.8939         | 0.9851         |
| DHIF-Net     | 3.616T    | 22.6695    | 47.8613     | 2.3849     | 0.9732      | 0.9948      | 47.7735        | 2.6987         | 0.8953         | 0.9855         |
| DSPNet       | 422.189G  | 6.0551     | 47.5665     | 2.5881     | 0.9722      | 0.9941      | 47.5973        | 2.7785         | 0.8938         | 0.9851         |
| MIMO-SST     | 98.1653G  | 4.983      | 48.3609     | 2.5834     | 0.9706      | 0.9944      | 47.6915        | 2.7804         | 0.8941         | 0.9852         |
| Mog-DCN      | 4.390T    | 7.071      | 48.5910     | 2.2541     | 0.9748      | 0.9953      | 47.8910        | 2.6964         | **0.8961**     | 0.9855         |
| LRTN         | 132.586G  | 3.535      | 47.7819     | 2.4387     | 0.9675      | 0.9939      | 47.7088        | 2.7330         | 0.8938         | 0.9855         |
| **Ours**     | **81.22G**| 1.33       | **49.3881** | **2.1816** | **0.9752**  | **0.9954**  | **47.8954**     | **2.6876**     | 0.8956         | **0.9856**     |

\* DAEM: 1.67G per fine-tuning, multiplied by the number of fine-tunings.


# Note： In the relearning process, I use a Transformer module composed of linear layers, without convolution operations.
