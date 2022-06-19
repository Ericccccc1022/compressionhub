import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
from torch.nn.parameter import Parameter
from .layers import *


class Encoder(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, img_size=224, patch_size=16, in_chans=3):
        super(Encoder, self).__init__()
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        middle_chans = 128
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, middle_chans, 5, stride=2, padding=2),   # in, out, k, s, padding
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(middle_chans, middle_chans, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(middle_chans, middle_chans, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(middle_chans, 192)
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)     # [HW/16**2, M], M=192
        return x


class Decoder_Reconstructor(nn.Module):
    '''
    Decode synthesis
    '''

    def __init__(self, img_size=224, patch_size=16, out_chans=3, embed_dim=768):
        super(Decoder_Reconstructor, self).__init__()
        embed_size = (img_size // patch_size, img_size // patch_size)
        num_patches = embed_size[0] * embed_size[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.num_patches = num_patches

        middle_chans = 128
        self.proj = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, middle_chans, 5, 2, output_padding=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(middle_chans, middle_chans, 5, 2, output_padding=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(middle_chans, middle_chans, 5, 2, output_padding=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(middle_chans, out_chans, 5, 2, output_padding=1, padding=2)
        )

    def forward(self, x):
        B, HW, C = x.shape
        x = self.proj(x.transpose(1, 2).reshape(B, C, self.embed_size[0], self.embed_size[1]))
        return x

class HyperEncoder(nn.Module):
    '''
    Analysis prior net
    '''
    def __init__(self, out_channel_N=128, out_channel_M=192):
        super(HyperEncoder, self).__init__()
        self.conv1 = nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.conv3 =  nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        # x = torch.abs(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv3(x)


class HyperDecoder(nn.Module):
    '''
    Decode synthesis prior
    '''
    def __init__(self, out_channel_N=128, out_channel_M=192):
        super(HyperDecoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N*3//2, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.deconv3 = nn.ConvTranspose2d(out_channel_N*3//2, out_channel_M*2, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.act(self.deconv1(x))
        x = self.act(self.deconv2(x))
        return self.deconv3(x)


class FFN(nn.Module):
    """
    FeedForward Network in ViT, part of Transformer Block
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.):   # dropout=0.1
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    Multi-Head Self-Attention Module in ViT, part of Transformer Block
    """
    def __init__(self, dim, num_heads=8, dropout=0.):   # dropout=0.1
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransfomerBlock(nn.Module):
    """
    Transformer Block in ViT
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0.):
        super(TransfomerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FFN(in_features=dim, hidden_features=mlp_hidden_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Rate_Distortion_Accuracy_Loss():
    def __init__(self, dim):
        super(Rate_Distortion_Accuracy_Loss, self).__init__()
        self.parametric_estimator = BitEstimator(dim)

    def cumulative(self, mu, sigma, x):
        """
        Calculates CDF of Normal distribution with parameters mu and sigma at point x
        """
        # half = 0.5
        # const = 2 ** -0.5
        # return half * (1 + torch.erf(const*(x-mu)/sigma))
        sigma = sigma.clamp(1e-10, 1e10)  # 方差
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)  # 为每个像素点创建独立的高斯分布
        return gaussian.cdf(x)

    def simple_cumulative(self, x):
        """
        Calculates CDF of Normal distribution with mu = 0 and sigma = 1
        """
        # half = 0.5
        # const = 2 ** -0.5
        # return half * (torch.erf(const * x)+1)
        mu = torch.zeros_like(x)
        sigma = torch.ones_like(x)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)  # 为每个像素点创建独立的高斯分布
        return gaussian.cdf(x)

    def latent_rate(self, mu, sigma, y):
        """
        Calculate latent rate
        Since we assume that each latent is modelled a Gaussian distribution convolved with Unit Uniform distribution we calculate latent rate
        as a difference of the CDF of Gaussian at two different points shifted by -1/2 and 1/2 (limit points of Uniform distribution)
        """
        upper = self.cumulative(mu, sigma, y + 0.5)
        lower = self.cumulative(mu, sigma, y - 0.5)
        return torch.sum(torch.clamp(-1.0 * torch.log(upper - lower + 1e-10) / math.log(2.0), 0, 50))


    def hyperlatent_rate_non_parametric(self, z):
        """
        Calculate hyperlatent rate
        Since we assume that each latent is modelled a Non-parametric convolved with Unit Uniform distribution we calculate latent rate
        as a difference of the CDF of the distribution at two different points shifted by -1/2 and 1/2 (limit points of Uniform distribution)
        """
        upper = self.simple_cumulative(z + .5)
        lower = self.simple_cumulative(z - .5)
        return torch.sum(torch.clamp(-1.0 * torch.log(upper - lower + 1e-10) / math.log(2.0), 0, 50))

    def hyperlatent_rate_parametric(self, z):
        """
        Calculate hyperlatent rate
        Since we assume that each latent is modelled a parametric convolved with Unit Uniform distribution we calculate latent rate
        as a difference of the CDF of the distribution at two different points shifted by -1/2 and 1/2 (limit points of Uniform distribution)
        """
        upper = self.parametric_estimator(z + .5)
        lower = self.parametric_estimator(z - .5)
        return torch.sum(torch.clamp(-1.0 * torch.log(upper - lower + 1e-10) / math.log(2.0), 0, 50))

    def rate_loss(self, x, mu, sigma, y_likelihood, z_likelihood):
        imgsize = x.shape[0] * x.shape[2] * x.shape[3]
        latent_rate = self.latent_rate(mu, sigma, y_likelihood) / imgsize
        hyperlatent_rate = self.hyperlatent_rate_parametric(z_likelihood) / imgsize

        loss = latent_rate + hyperlatent_rate
        return loss, latent_rate, hyperlatent_rate

    def distortion_accuracy_loss(self, x, x_rec, x_cls, label, alpha=1, beta=0.01):
        criterion = torch.nn.CrossEntropyLoss()  # 定义损失
        mse = torch.mean((x - x_rec).pow(2))
        accloss = criterion(x_cls, label)
        return alpha * accloss + beta * mse, mse, accloss


class Transformer21(nn.Module):
    """
    Joint Compression and Classification Transformer
    Tip: For the convenience of "feature fusion", we must define Decoder_Classifier heads within this class
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., drop_rate=0.):
        super(Transformer21, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.depth = depth

        # 1.Encoding Module
        # encoding into [H/16, W/16, M],  M=192
        self.encoder = Encoder(img_size, patch_size, in_chans)   # embed_dim=C
        # expand the channel into [HW/16**2, C],  C=768
        self.chans_convert = nn.Linear(192, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # 2.Hyper Module
        self.hyper_encoder = HyperEncoder()
        self.hyper_decoder = HyperDecoder()

        # 3.Decoding Module
        # Transformer block
        self.blocks = nn.ModuleList([
            TransfomerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=drop_rate)
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier head
        self.head_cls = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Reconstruction head
        self.head_rec = Decoder_Reconstructor(img_size=img_size, patch_size=patch_size, out_chans=in_chans, embed_dim=embed_dim)

        # fusion
        self.fusion0 = nn.Linear(embed_dim, embed_dim//4)
        self.fusion1 = nn.Linear(embed_dim, embed_dim // 4)
        self.fusion2 = nn.Linear(embed_dim, embed_dim // 4)
        self.fusion3 = nn.Linear(embed_dim, embed_dim // 4)
        self.fusion = nn.Linear(embed_dim, embed_dim)

        # 4.Loss
        self.rdloss = Rate_Distortion_Accuracy_Loss(192)


    def quantize(self, x):
        uniform_noise = torch.nn.init.uniform_(torch.empty_like(x), -0.5, 0.5).cuda()
        # For inference
        y_hat = torch.round(x)
        # For training
        y_likelihood = x + uniform_noise

        return y_hat, y_likelihood


    def forward(self, x, label):    # need GT label to calculate acc_loss
        y_patch = self.encoder(x)     # [HW/16**2, 192]
        B, N, C = y_patch.shape
        H = W = int(N ** 0.5)

        y = y_patch.transpose(1, 2).reshape(B, C, H, W)     # [B, C, H, W]
        y_hat, y_likelihood = self.quantize(y)
        z = self.hyper_encoder(y)
        z_hat, z_likelihood = self.quantize(z)
        if self.training:
            musigma = self.hyper_decoder(z_likelihood)
        else:
            musigma = self.hyper_decoder(z_hat)
        mu, sigma = torch.split(musigma, y_likelihood.shape[1], dim=1)
        # sigma = torch.clamp(sigma, min=1e-8)

        # clipped_x_hat = x_hat.clamp(0., 1.)
        # Calculate Rate-Loss Now
        # todo:问题：如果先算rdloss，没有fusion情况下不知道x_hat；如果最后算rdloss，那么y_likelihood已经发生改变
        # todo:解决方法：设计aux_loss()，先计算y和z的rate loss
        bpp, bpp_y, bpp_z = self.rdloss.rate_loss(x, mu, sigma, y_likelihood, z_likelihood)


        # Transfomrer
        y_likelihood = y_likelihood.flatten(2).transpose(1, 2)  # [B, HW/16**2, 192]
        y_likelihood = self.chans_embed(y_likelihood)   # [B, HW/16**2, 768]
        cls_tokens = self.cls_token.expand(B, -1, -1)     # [1, 1, 768] --> [B, 1, 768]
        y0 = torch.cat((cls_tokens, y_hat), dim=1)

        y0 = self.pos_drop(y0)
        y1 = self.blocks[0](y0)
        y2 = self.blocks[1](y1)
        y3 = self.blocks[2](y2)
        y4 = self.blocks[3](y3)
        y5 = self.blocks[4](y4)
        y6 = self.blocks[5](y5)
        y7 = self.blocks[6](y6)
        y8 = self.blocks[7](y7)
        y9 = self.blocks[8](y8)
        y10 = self.blocks[9](y9)
        y11 = self.blocks[10](y10)
        y12 = self.blocks[11](y11)
        y_out = self.norm(y12)

        y0 = self.fusion0(y_likelihood)
        y1 = self.fusion1(y1[:, 1:])
        y2 = self.fusion2(y2[:, 1:])
        y3 = self.fusion3(y3[:, 1:])

        y_rec = torch.cat((y0, y1, y2, y3), dim=2)
        y_rec = self.fusion(y_rec)

        x_rec = self.head_rec(y_rec)
        x_cls = self.head_cls(y_out[:, 0])


        # Calculate Distortion-Accuracy Loss
        dist_acc_loss, mse, accloss = self.rdloss.distortion_accuracy_loss(x, x_rec, x_cls, label, alpha=1, beta=0.01)

        return dist_acc_loss + bpp, dist_acc_loss, mse, bpp, bpp_y, bpp_z







