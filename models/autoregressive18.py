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
    def __init__(self, dim_in, dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, dim, 5, 2)   # kernel_size=5, stride=2
        self.conv2 = nn.Conv2d(dim, dim, 5, 2)
        self.conv3 = nn.Conv2d(dim, dim, 5, 2)
        self.conv4 = nn.Conv2d(dim, dim, 5, 2)
        self.gdn = GDN(dim)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, gain=1)
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, gain=1)
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, gain=1)
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, gain=1)
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)

    def forward(self, x):
        x = self.gdn(self.conv1(x))
        x = self.gdn(self.conv2(x))
        x = self.gdn(self.conv3(x))
        x = self.conv4(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dim_in, dim):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(dim_in, dim, 5, 2)
        self.deconv2 = nn.ConvTranspose2d(dim, dim, 5, 2)
        self.deconv3 = nn.ConvTranspose2d(dim, dim, 5, 2, output_padding=1)    # out_padding = 1
        self.last_deconv = nn.ConvTranspose2d(dim, 3, 5, 2, output_padding=1)  # out_padding = 1
        self.igdn = GDN(dim, inverse=True)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, gain=1)
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, gain=1)
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, gain=1)
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.last_deconv.weight.data, gain=1)
        torch.nn.init.constant_(self.last_deconv.bias.data, 0.01)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.igdn(x)
        x = self.deconv2(x)
        x = self.igdn(x)
        x = self.deconv3(x)
        x = self.igdn(x)
        x = self.last_deconv(x)
        return x


class HyperEncoder(nn.Module):
    def __init__(self, dim_in, dim):
        super(HyperEncoder, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, dim, 3, 1)
        self.conv2 = nn.Conv2d(dim, dim, 5, 2, 1)
        self.conv3 = nn.Conv2d(dim, dim, 5, 2)
        self.act = nn.LeakyReLU()
        torch.nn.init.xavier_normal_(self.conv1.weight.data, gain=1)
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, gain=1)
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, gain=1)
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class HyperDecoder(nn.Module):
    def __init__(self, dim_in, dim):
        super(HyperDecoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(dim_in, dim, 5, 2)
        self.deconv2 = nn.ConvTranspose2d(dim, dim*3//2, 5, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(dim*3//2, dim*2, 3, 1)
        self.act = nn.LeakyReLU()
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, gain=1)
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, gain=1)
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, gain=1)
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)

    def forward(self, x):
        x = self.act(self.deconv1(x))
        x = self.act(self.deconv2(x))
        x = self.deconv3(x)
        return x


class ContextPrediction(nn.Module):
    def __init__(self, dim_in, dim):
        super(ContextPrediction, self).__init__()
        self.masked = MaskConv2d("A", dim_in, dim*2, 5, 1, 2)

    def forward(self, x):
        return self.masked(x)


class EntropyParameters(nn.Module):
    def __init__(self, dim_in, dim):
        super(EntropyParameters, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, 640, 1, 1)   # 192 640 512 384
        self.conv2 = nn.Conv2d(640, 512, 1, 1)
        self.conv3 = nn.Conv2d(512, dim*2, 1, 1)
        self.act = nn.LeakyReLU()
        torch.nn.init.xavier_normal_(self.conv1.weight.data, gain=1)
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, gain=1)
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, gain=1)
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class RateDistortionLoss(nn.Module):
    def __init__(self, dim):
        super(RateDistortionLoss, self).__init__()
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
            loss = lam * (1.0 - msssim) + latent_rate + hyperlatent_rate
            return loss, msssim, latent_rate, hyperlatent_rate
        # Optimized for MSE
        else:
            mse = torch.mean((x - x_hat).pow(2))
            loss = lam * mse + latent_rate + hyperlatent_rate
            return loss, mse, latent_rate, hyperlatent_rate


class Autoregressive18(nn.Module):
    def __init__(self, dim):
        super(Autoregressive18, self).__init__()
        self.Encoder = Encoder(3, dim)
        self.Decoder = Decoder(dim, dim)
        self.hyper_encoder = HyperEncoder(dim, dim)
        self.hyper_decoder = HyperDecoder(dim, dim)
        self.entropy = EntropyParameters(dim*4, dim)
        self.context = ContextPrediction(dim, dim)
        self.rdloss = RateDistortionLoss(dim)

    def quantize(self, x):
        uniform_noise = torch.nn.init.uniform_(torch.empty_like(x), -0.5, 0.5).cuda()
        # For inference
        y_hat = torch.round(x)
        # For training
        y_likelihood = x + uniform_noise

        return y_hat, y_likelihood

    def forward(self, x, lam=0.0003, use_ssim=False):
        y = self.Encoder(x)
        y_hat, y_likelihood = self.quantize(y)
        z = self.hyper_encoder(y)
        z_hat, z_likelihood = self.quantize(z)
        if self.training:
            phi = self.context(y_likelihood)
            psi = self.hyper_decoder(z_likelihood)
            x_hat = self.Decoder(y_likelihood)
        else:
            phi = self.context(y_hat)
            psi = self.hyper_decoder(z_hat)
            x_hat = self.Decoder(y_hat)
        phi_psi = torch.cat([phi, psi], dim=1)  # concat along channels
        sigma_mu = self.entropy(phi_psi)
        mu, sigma = torch.split(sigma_mu, y_likelihood.shape[1], dim=1)
        clipped_x_hat = x_hat.clamp(0., 1.)

        # Calculate Loss Now
        loss, distortion, bpp_y, bpp_z = self.rdloss(x, clipped_x_hat, mu, sigma, y_likelihood, z_likelihood, lam, use_ssim)
        bpp = bpp_y + bpp_z

        return clipped_x_hat, loss, distortion, bpp_y, bpp_z, bpp