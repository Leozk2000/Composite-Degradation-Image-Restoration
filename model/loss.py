import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from math import exp
from torchvision import transforms
from torchvision.models import vgg16
import torchvision
'''
MS-SSIM Loss
'''

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(img1.device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
 
    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])  #算出总共求了多少次差
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()  
        # x[:,:,1:,:]-x[:,:,:h_x-1,:]就是对原图进行错位，分成两张像素位置差1的图片，第一张图片
        # 从像素点1开始（原图从0开始），到最后一个像素点，第二张图片从像素点0开始，到倒数第二个            
        # 像素点，这样就实现了对原图进行错位，分成两张图的操作，做差之后就是原图中每个像素点与相
        # 邻的下一个像素点的差。
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class ContrastLoss(nn.Module):
    def __init__(self):
        super(ContrastLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.model = vgg16(weights = torchvision.models.VGG16_Weights.DEFAULT)
        self.model = self.model.features[:16].to("cuda" if torch.cuda.is_available() else "cpu")
        for param in self.model.parameters():
            param.requires_grad = False
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def gen_features(self, x):
        output = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output
    def forward(self, inp, pos, neg, out):
        inp_t = inp
        inp_x0 = self.gen_features(inp_t)
        pos_t = pos
        pos_x0 = self.gen_features(pos_t)
        out_t = out
        out_x0 = self.gen_features(out_t)
        neg_t, neg_x0 = [],[]
        for i in range(neg.shape[1]):
            neg_i = neg[:,i,:,:]
            neg_t.append(neg_i)
            neg_x0_i = self.gen_features(neg_i)
            neg_x0.append(neg_x0_i)
        loss = 0
        for i in range(len(pos_x0)):
            pos_term = self.l1(out_x0[i], pos_x0[i].detach())
            inp_term = self.l1(out_x0[i], inp_x0[i].detach())/(len(neg_x0)+1)
            neg_term = sum(self.l1(out_x0[i], neg_x0[j][i].detach()) for j in range(len(neg_x0)))/(len(neg_x0)+1)
            loss = loss + pos_term / (inp_term+neg_term+1e-7)
        return loss / len(pos_x0)

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x.to('cuda:0') - y.to('cuda:0')
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

# class EdgeLoss(nn.Module):
#     def __init__(self):
#         super(EdgeLoss, self).__init__()
#         k = torch.Tensor([[.05, .25, .4, .25, .05]])
#         self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
#         if torch.cuda.is_available():
#             self.kernel = self.kernel.to('cuda:0')
#         self.loss = CharbonnierLoss()

#     def conv_gauss(self, img):
#         n_channels, _, kw, kh = self.kernel.shape
#         img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
#         return F.conv2d(img, self.kernel, groups=n_channels)

#     def laplacian_kernel(self, current):
#         filtered    = self.conv_gauss(current)
#         down        = filtered[:,:,::2,::2]
#         new_filter  = torch.zeros_like(filtered)
#         new_filter[:,:,::2,::2] = down*4
#         filtered    = self.conv_gauss(new_filter)
#         diff = current - filtered
#         return diff

#     def forward(self, x, y):
#         loss = self.loss(self.laplacian_kernel(x.to('cuda:0')), self.laplacian_kernel(y.to('cuda:0')))
#         return loss

# class fftLoss(nn.Module):
#     def __init__(self):
#         super(fftLoss, self).__init__()

#     def forward(self, x, y):
#         diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
#         loss = torch.mean(abs(diff))
#         return loss


class Total_loss(nn.Module):
    def __init__(self, args):
        super(Total_loss, self).__init__()
        # Loss components
        self.msssim_loss = MSSSIM()
        self.contrast_loss = ContrastLoss()
        self.charbonnier_loss = CharbonnierLoss()

        # Move all loss modules to GPU if available
        if torch.cuda.is_available():
            self.msssim_loss = self.msssim_loss.cuda()
            self.contrast_loss = self.contrast_loss.cuda()
            self.charbonnier_loss = self.charbonnier_loss.cuda()
        
        # Unpack weights from args.loss_weight tuple
        (self.weight_smooth_l1,    # 0.5  (main reconstruction)
         self.weight_msssim,       # 0.3  (structural similarity)
         self.weight_contrast,     # 0.15  (contrastive learning)
         self.weight_charbonnier   # 0.05  (deblurring)
         ) = args.loss_weight

    def _move_to_device(self, x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, (list, tuple)):
            return [self._move_to_device(item, device) for item in x]
        return x

    def forward(self, inp, pos, neg, out):
        # Ensure all inputs are on the same device
        device = out.device
        pos = pos.to(device)
        
        out = torch.clamp(out, 0, 1)
        pos = torch.clamp(pos, 0, 1)

        # Basic reconstruction loss
        smooth_l1_loss = F.smooth_l1_loss(out, pos) * self.weight_smooth_l1
        
        # Structural similarity loss
        msssim_loss = (1 - self.msssim_loss(out, pos)) * self.weight_msssim
        
        # Optional losses based on image quality
        contrast_loss = self.contrast_loss(inp[0], pos, neg, out) * self.weight_contrast
        charbonnier_loss = self.charbonnier_loss(out, pos) * self.weight_charbonnier
        
        total_loss = smooth_l1_loss + msssim_loss + contrast_loss + charbonnier_loss
        
        # Debug info
        if total_loss.item() > 10:
            print(f"High loss detected: {total_loss.item()}")
            print(f"Components: L1={smooth_l1_loss.item():.4f}, "
                  f"MSSSIM={msssim_loss.item():.4f}, "
                  f"Contrast={contrast_loss.item():.4f}, "
                  f"Char={charbonnier_loss.item():.4f}")
        
        return total_loss