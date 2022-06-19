import numpy as np
import os
import random
import sympy
from scipy.special import gamma
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
from torch.nn.parameter import Parameter
from .layers import GDN, BitEstimator

# T in the paper
def quality2lambda(qmap):
    return 1e-3 * torch.exp(4.382 * qmap)


