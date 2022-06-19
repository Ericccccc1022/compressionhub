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


class Analysis_net(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Analysis_net, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(out_channel_N)
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(out_channel_N)
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(out_channel_N)
        self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        return self.conv4(x)


class Analysis_prior_net(nn.Module):
    '''
    Analysis prior net
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Analysis_prior_net, self).__init__()
        self.conv1 = nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = nn.ReLU()
        self.conv3 =  nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)

    def forward(self, x):
        x = torch.abs(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.conv3(x)


class Synthesis_net(nn.Module):
    '''
    Decode synthesis
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Synthesis_net, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(out_channel_N, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(out_channel_N, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.igdn3 = GDN(out_channel_N, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(out_channel_N, 3, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, (math.sqrt(2 * 1 * (out_channel_N + 3) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)

    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        x = self.deconv4(x)
        return x


class Synthesis_prior_net(nn.Module):
    '''
    Decode synthesis prior
    '''
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Synthesis_prior_net, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.relu2 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        return torch.exp(self.deconv3(x))



class RateDistortionLoss(nn.Module):
    def __init__(self, out_channel_N):
        super(RateDistortionLoss, self).__init__()
        self.parametric_estimator = BitEstimator(out_channel_N)

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

    def forward(self, x, x_hat, mu, sigma, y_lieklihood, z_likelihood, lam, use_ssim=False):
        """
        Calculate Rate-Distortion Loss
        """
        imgsize = x.shape[0] * x.shape[2] * x.shape[3]
        latent_rate = self.latent_rate(mu, sigma, y_lieklihood) / imgsize

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


class Hyperprior18(nn.Module):
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(Hyperprior18, self).__init__()
        self.Encoder = Analysis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.Decoder = Synthesis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorEncoder = Analysis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorDecoder = Synthesis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M
        self.rdloss = RateDistortionLoss(out_channel_N)

    def quantize(self, x):
        uniform_noise = torch.nn.init.uniform_(torch.empty_like(x), -0.5, 0.5).cuda()
        # For inference
        y_hat = torch.round(x)
        # For training
        y_likelihood = x + uniform_noise

        return y_hat, y_likelihood

    def forward(self, x, lam, use_ssim=False):
        y = self.Encoder(x)     # y
        z = self.priorEncoder(y)  # z
        y_hat, y_likelihood = self.quantize(y)
        z_hat, z_likelihood = self.quantize(z)
        if self.training:
            x_hat = self.Decoder(y_likelihood)
            sigma = self.priorDecoder(z_likelihood)
        else:
            x_hat = self.Decoder(y_hat)
            sigma = self.priorDecoder(z_hat)

        clipped_x_hat = x_hat.clamp(0., 1.)
        # sigma = torch.clamp(sigma, min=1e-8)

        # Calculate Loss Now
        mu = torch.zeros_like(sigma)
        loss, distortion, bpp_y, bpp_z = self.rdloss(x, clipped_x_hat, mu, sigma, y_likelihood, z_likelihood, lam, use_ssim)
        bpp = bpp_y + bpp_z

        return clipped_x_hat, loss, distortion, bpp_y, bpp_z, bpp