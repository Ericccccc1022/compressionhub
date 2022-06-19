import numpy as np
import os
import random
import sympy
from scipy.special import gamma
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

class RB(nn.Module):
    """
    Residual Block: The basic part of the Attention Module
    In the origin paper, it goes through 3x3conv --> ReLU --> 3x3conv
    The problem remained is that: the channels always be the same
    """
    def __init__(self, dim):
        super(RB, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, stride=1, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return out + x


class NLB(nn.Module):
    """
    Non-local Block: The main part of the Attention Module
    Compared to 'self-attention', it adds the input to the final result
    """
    def __init__(self, dim):
        super(NLB, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim//2, 1, stride=1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1, stride=1)
        self.conv3 = nn.Conv2d(dim, dim//2, 1, stride=1)
        self.lastconv = nn.Conv2d(dim//2, dim, 1, stride=1)

    def forward(self, x):
        out1 = self.conv1(x)    # [B, C, H, W]
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        B, C, H, W = out1.shape

        out1 = out1.contiguous().view(-1, out1.shape[1])  # [BHW, C]
        out2 = out2.contiguous().view(-1, out2.shape[1]).transpose(0, 1)  # [C, BHW]
        out3 = out3.contiguous().view(-1, out3.shape[1])  # [BHW, C]
        out = torch.matmul(out1, out2)  # [BHW, BHW]
        out = F.softmax(out)

        out = torch.matmul(out, out3)   # [BHW, C]
        out = out.contiguous().view(B, C, H, W)
        out = self.lastconv(out)    # [B, H, W, 2C]

        return out + x


class Attention(nn.Module):
    """
    Attention Module proposed in "Residual Non-Local Attention Networks For Image Restoration"
    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.rb1 = RB(dim)
        self.rb2 = RB(dim)
        self.rb3 = RB(dim)
        self.rb4 = RB(dim)
        self.rb5 = RB(dim)
        self.rb6 = RB(dim)
        self.lastconv = nn.Conv2d(dim, dim, 1, stride=1)
        # self.nlb = NLB(dim)

    def forward(self, x):
        local = self.rb3(self.rb2(self.rb1(x)))
        # nlocal = self.nlb(x)
        # nlocal = self.rb6(self.rb5(self.rb4(nlocal)))
        nlocal = self.rb6(self.rb5(self.rb4(x)))
        nlocal = self.lastconv(nlocal)
        nlocal = torch.sigmoid(nlocal)
        out = local * nlocal
        return out + x


class Encoder(nn.Module):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(dim, 192, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(192, 192, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(192, 192, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(192, 192, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 1, stride=1)
        self.attention = Attention(192)
        self.gdn = GDN(192)
        self.gain_unit = nn.Parameter(torch.nn.init.normal_(torch.empty(6, 192), 0, 0.01))    # [n, 192]  论文中设置n=6

    def forward(self, x, s, l, s1, s2):
        x = self.gdn(self.conv1(x))
        x = self.attention(x)
        x = self.gdn(self.conv2(x))
        x = self.gdn(self.conv3(x))
        x = self.conv4(x)
        x = self.conv5(x)
        B, C, H, W = x.shape

        # todo:Gain unit
        if self.training:
            self.gain_vector = abs(self.gain_unit[s, :])
        else:
            # self.gain_vector = (abs(self.gain_unit[s1, :]) ** l) * (abs(self.gain_unit[s2, :]) ** (1-l)) # Actually, "l" should be controlled manually
            self.gain_vector = (self.gain_unit[s1, :] ** l) * (self.gain_unit[s2, :] ** (1 - l))
            # print('shape:', self.gain_unit.shape)
            # print('vector0:', self.gain_unit[0,:])
            # print('vector1:', self.gain_unit[1,:])
            # print('vector01:', self.gain_vector)
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
        x = x * self.gain_vector
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        return x


class Decoder(nn.Module):
    def __init__(self, dim):
        super(Decoder, self).__init__()
        self.rb1 = RB(dim)
        self.rb2 = RB(dim)
        self.deconv1 = nn.ConvTranspose2d(dim, 192, 3, 2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(192, 192, 3, 2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(192, 192, 3, 2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(192, 3, 3, 2, padding=1, output_padding=1)
        self.attention = Attention(192)
        self.igdn = GDN(192, inverse=True)
        self.inverse_gain_unit = nn.Parameter(torch.nn.init.normal_(torch.empty(6, 192), 0, 0.01))    # [n, 192]

    def forward(self, x, s, l, s1, s2):
        B, C, H, W = x.shape
        # todo:Inverse Gain unit
        if self.training:
            self.inverse_gain_vector = abs(self.inverse_gain_unit[s, :])
        else:
            # self.inverse_gain_vector = (abs(self.inverse_gain_unit[s1, :]) ** l) * (abs(self.inverse_gain_unit[s2, :]) ** (1-l)) # Actually, "l" should be controlled manually
            self.inverse_gain_vector = (self.inverse_gain_unit[s1, :] ** l) * (self.inverse_gain_unit[s2, :] ** (1 - l))
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
        x = x * self.inverse_gain_vector
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        x = self.rb1(x)
        x = self.rb2(x)
        x = self.igdn(self.deconv1(x))
        x = self.igdn(self.deconv2(x))
        x = self.deconv3(x)
        x = self.attention(x)
        x = self.igdn(x)
        x = self.deconv4(x)
        return x

class HyperEncoder(nn.Module):
    def __init__(self, dim):
        super(HyperEncoder, self).__init__()
        self.conv1 = nn.Conv2d(dim, 192, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(192, 192, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(192, 192, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(192, 192, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, stride=2, padding=1)
        self.gain_unit = nn.Parameter(torch.nn.init.normal_(torch.empty(6, 192), 0, 0.01))  # [n, 192]

    def forward(self, x, s, l, s1, s2):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv3(self.conv2(x)))
        x = self.conv5(self.conv4(x))
        B, C, H, W = x.shape
        # todo:Gain unit
        if self.training:
            self.gain_vector = abs(self.gain_unit[s, :])
        else:
            self.gain_vector = (abs(self.gain_unit[s1, :]) ** l) * (abs(self.gain_unit[s2, :]) ** (1-l))  # Actually, "l" should be controlled manually
            self.gain_vector = (self.gain_unit[s1, :] ** l) * (self.gain_unit[s2, :] ** (1 - l))  # Actually, "l" should be controlled manually
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
        x = x * self.gain_vector
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        return x

class HyperDecoder(nn.Module):
    def __init__(self, dim):
        super(HyperDecoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(dim, 192, 3, 1, padding=1)   # 1
        self.deconv2 = nn.ConvTranspose2d(192, 192, 3, 2, padding=1, output_padding=1)   # up2
        self.deconv3 = nn.ConvTranspose2d(192, 192, 3, 1, padding=1)   # 1
        self.deconv4 = nn.ConvTranspose2d(192, 288, 3, 2, padding=1, output_padding=1)   # up2
        self.deconv5 = nn.ConvTranspose2d(288, 384, 3, 1, padding=1)   # 1
        self.inverse_gain_unit = nn.Parameter(torch.nn.init.normal_(torch.empty(6, 192), 0, 0.01))

    def forward(self, x, s, l, s1, s2):
        B, C, H, W = x.shape
        # todo:Inverse Gain unit
        # Divided into two streams: 1)training  2)inference: exponential interpolation
        if self.training:
            self.inverse_gain_vector = abs(self.inverse_gain_unit[s, :])
        else:
            # self.inverse_gain_vector = (abs(self.inverse_gain_unit[s1, :]) ** l) * (abs(self.inverse_gain_unit[s2, :]) ** (1-l))  # Actually, "l" should be controlled manually
            self.inverse_gain_vector = (self.inverse_gain_unit[s1, :] ** l) * (self.inverse_gain_unit[s2, :] ** (1 - l))
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
        x = x * self.inverse_gain_vector
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        x = F.leaky_relu(self.deconv2(self.deconv1(x)))
        x = F.leaky_relu(self.deconv4(self.deconv3(x)))
        x = F.leaky_relu(self.deconv5(x))
        return x

class ContextPrediction(nn.Module):
    def __init__(self, dim_in):
        super(ContextPrediction, self).__init__()
        self.masked1 = MaskConv2d("A", dim_in, 384, 3, 1, 1)
        self.masked2 = MaskConv2d("A", dim_in, 384, 5, 1, 2)
        self.masked3 = MaskConv2d("A", dim_in, 384, 7, 1, 3)

    def forward(self, x):
        out1 = self.masked1(x)
        out2 = self.masked2(x)
        out3 = self.masked3(x)
        out = torch.cat([out1, out2, out3], dim=1)    # concatenate along channels
        return out

class EntropyParameters(nn.Module):
    def __init__(self, dim):
        super(EntropyParameters, self).__init__()
        self.conv1 = nn.Conv2d(dim, 1280, 1, 1)
        self.conv2 = nn.Conv2d(1280, 1024, 1, 1)
        self.conv3 = nn.Conv2d(1024, 768, 1, 1)
        self.conv4 = nn.Conv2d(768, 640, 1, 1)
        self.conv5 = nn.Conv2d(640, 576, 1, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = self.conv5(x)
        return x    # channels:576 = 3*192(mean, left scale, right scale)


class RateDistortionLoss(nn.Module):
    def __init__(self):
        super(RateDistortionLoss, self).__init__()
        self.parametric_estimator = BitEstimator(192)

    def AGD_cdf(self, x, alpha, sigma_r, sigma_l):
        """
        CDF of the Asymmetric Gaussian Distribution used in this paper!
        """
        t = sympy.symbols("t")
        coef = alpha / (sigma_l * sigma_r * gamma(1 / alpha))
        coef_exp_l = -1. * (-t / sigma_l) ** alpha
        coef_exp_r = -1. * (-t / sigma_r) ** alpha
        if x < 0:
            cdf = sympy.integrate(coef * sympy.exp(coef_exp_l), (t, -20000, x))
            return float(cdf)
        else:
            cdf_l = sympy.integrate(coef * sympy.exp(coef_exp_l), (t, -20000, 0))
            cdf_r = sympy.integrate(coef * sympy.exp(coef_exp_r), (t, 0, x))
            return float(cdf_l) + float(cdf_r)

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

    def forward(self, x, x_hat, mu, sigma, y_likelihood, z_likelihood, lam, use_ssim=False):
        """
        Calculate Rate-Distortion Loss
        """
        imgsize = x.shape[0] * x.shape[2] * x.shape[3]
        latent_rate = self.latent_rate(mu, sigma, y_likelihood) / imgsize

        # two options to calculate entropy of z:
        # 1. non-parametric
        # hyperlatent_rate = self.hyperlatent_rate(z_likelihood) / imgsize
        # 2. parametric (Appendix6.1)
        hyperlatent_rate = self.hyperlatent_rate_parametric(z_likelihood) / imgsize

        # Optimized for MS-SSIM
        if use_ssim:
            msssim = ms_ssim(x_hat, x, data_range=1.0, size_average=True)
            # loss = 1.0 - msssim + lam * (latent_rate + hyperlatent_rate)
            loss = 1.0 - msssim + lam * (latent_rate + hyperlatent_rate)
            return loss, msssim, latent_rate, hyperlatent_rate
        # Optimized for MSE
        else:
            mse = torch.mean((x - x_hat).pow(2))
            loss = mse + lam * (latent_rate + hyperlatent_rate)
            return loss, mse, latent_rate, hyperlatent_rate


class Asymmetric21(nn.Module):
    def __init__(self):
        super(Asymmetric21, self).__init__()
        self.Encoder = Encoder(3)
        self.Decoder = Decoder(192)
        self.hyper_encoder = HyperEncoder(192)
        self.hyper_decoder = HyperDecoder(192)
        self.entropy = EntropyParameters(384*4)  # 1536
        self.context = ContextPrediction(192)
        self.rdloss = RateDistortionLoss()
        self.B_mse = [0.05, 0.03, 0.007, 0.003, 0.001, 0.0003]      # variable Lagrange multipliers
        self.B_msssim = [0.07, 0.03, 0.007, 0.003, 0.001, 0.0006]
        self.cal_s = []

    def quantize(self, x):
        uniform_noise = torch.nn.init.uniform_(torch.empty_like(x), -0.5, 0.5).cuda()
        # For inference
        y_hat = torch.round(x)
        # For training
        y_likelihood = x + uniform_noise

        return y_hat, y_likelihood

    def forward(self, x, l, s1=0, s2=0, use_ssim=False):
        s = random.randint(0, len(self.B_msssim)-1)
        self.cal_s.append(s)
        if use_ssim:
            lam = self.B_msssim[s]  # change lambda according to s
        else:
            lam = self.B_mse[s]  # change lambda according to s

        y = self.Encoder(x, s, l, s1, s2)
        # print('y:', y[0,0,:,:])
        y_hat, y_likelihood = self.quantize(y)
        z = self.hyper_encoder(y, s, l, s1, s2)
        z_hat, z_likelihood = self.quantize(z)
        if self.training:
            phi = self.context(y_likelihood)
            psi = self.hyper_decoder(z_likelihood, s, l, s1, s2)
        else:
            phi = self.context(y_hat)
            psi = self.hyper_decoder(z_hat, s, l, s1, s2)
        phi_psi = torch.cat([phi, psi], dim=1)  # concat along channels
        sigma_mu = self.entropy(phi_psi)    # [b, 576, H, W]
        sigma_l, sigma_r, mu = torch.split(sigma_mu, y_likelihood.shape[1], dim=1)     # split into 3 parts: 576 = 192 * 3(mean, left scale, right scale)
        x_hat = self.Decoder(y_likelihood, s, l, s1, s2)
        clipped_x_hat = x_hat.clamp(0., 1.)
        # print('x_hat', clipped_x_hat)
        # print('mse:', torch.mean((x - clipped_x_hat).pow(2)))
        # print('ssim:', ms_ssim(clipped_x_hat, x, data_range=1.0, size_average=True))
        # print('isnan:', torch.isnan(clipped_x_hat).any())
        
        # Calculate Loss Now
        loss, distortion, bpp_y, bpp_z = self.rdloss(x, clipped_x_hat, mu, sigma_l, y_likelihood, z_likelihood, lam, use_ssim)    # todo:3.修改为非对称高斯
        bpp = bpp_y + bpp_z

        return clipped_x_hat, loss, distortion, bpp_y, bpp_z, bpp, self.cal_s
