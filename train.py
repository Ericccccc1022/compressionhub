import os
import argparse
from models import *
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
import json
import time
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from utils import *


# Some default settings
torch.backends.cudnn.enabled = True
logger = logging.getLogger("ImageCompression")  # Build logger


def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch reimplement')
    parser.add_argument('--algorithm', default='hyperprior18', help='experiment name')
    parser.add_argument('--name', default='hyperprior18_16_100e', help='experiment name')
    parser.add_argument('--log_name', default='log_16', help='experiment name')
    # parser.add_argument('--pretrain', default='./checkpoints/hyperprior18_16_100e/epoch_14.pth.tar', help='load pretrain model')
    parser.add_argument('--pretrain', default='', help='load pretrain model')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--config', dest='config', default='./configs/hyperprior_ssim_16.json', help='hyperparameter in json format')
    parser.add_argument('--seed', default=10, type=int, help='seed for random functions, and network initialization')
    parser.add_argument('--train', dest='train', default='./data/coco2017/train2017/', help='the path of training dataset')
    parser.add_argument('--val', dest='val', required=False, help='the path of validation dataset')

    parser.add_argument('--mode', default='SSIM', help='experiment name')
    parser.add_argument('--gpu_num', default=1, help='experiment name')
    parser.add_argument('--base_lr', default=1e-4, help='experiment name')
    parser.add_argument('--cur_lr', default=1e-4, help='experiment name')
    parser.add_argument('--train_lambda', default=16, help='experiment name')
    parser.add_argument('--cal_step', default=50, help='experiment name')
    parser.add_argument('--print_freq', default=100, help='experiment name')
    parser.add_argument('--warmup_step', default=0, help='experiment name')
    parser.add_argument('--batch_size', default=8, help='experiment name')
    parser.add_argument('--tot_epoch', default=100, help='experiment name')
    parser.add_argument('--tot_step', default=None, help='experiment name')
    parser.add_argument('--decay_interval', default=60, help='experiment name')
    parser.add_argument('--lr_decay', default=0.1, help='experiment name')
    parser.add_argument('--image_size', default=256, help='experiment name')
    parser.add_argument('--global_step', default=0, help='experiment name')
    parser.add_argument('--out_channel_N', default=192, help='experiment name')
    parser.add_argument('--out_channel_M', default=320, help='experiment name')
    parser.add_argument('--out_channel', default=192, help='experiment name')
    parser.add_argument('--device', default='cuda', help='experiment name')
    args = parser.parse_args()

    assert args.algorithm in ['hyperprior18', 'autoregressive18', 'asymmetric21', 'transformer21']
    return args

def parse_config(args):
    config = json.load(open(args.config))
    if 'tot_epoch' in config:
        args.tot_epoch = config['tot_epoch']
    if 'tot_step' in config:
        args.tot_step = config['tot_step']
    if 'train_lambda' in config:
        args.train_lambda = config['train_lambda']
    if 'batch_size' in config:
        args.batch_size = config['batch_size']
    if "print_freq" in config:
        args.print_freq = config['print_freq']
    if "mode" in config:
        args.mode = config['mode']
    if 'lr' in config:
        if 'base' in config['lr']:
            args.base_lr = config['lr']['base']
            args.cur_lr = args.base_lr
        if 'decay' in config['lr']:
            args.lr_decay = config['lr']['decay']
        if 'decay_interval' in config['lr']:
            args.decay_interval = config['lr']['decay_interval']
    if 'out_channel_N' in config:
        args.out_channel_N = config['out_channel_N']
    if 'out_channel_M' in config:
        args.out_channel_M = config['out_channel_M']

def adjust_learning_rate(args, optimizer, global_step):
    """
    warm up the learning rate
    """
    if args.global_step < args.warmup_step:
        lr = args.base_lr * args.global_step / args.warmup_step
    elif global_step < (args.decay_interval * (len(train_dataset)//args.batch_size)):
        lr = args.base_lr
    else:
        lr = args.base_lr * args.lr_decay
    args.cur_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, args):
    net.train()
    elapsed, losses, psnrs, bpps, bpp_features, bpp_zs, Distortion = [AverageMeter() for _ in range(7)]   # print every 100 iters
    loop = tqdm(train_loader, total=len(train_dataset) // args.batch_size)
    for epoch in range(args.tot_epoch):
        logger.info("Epoch {} begin".format(epoch))
        adjust_learning_rate(args, optimizer, args.global_step)    # update lr first

        for input in loop:
            start_time = time.time()
            args.global_step += 1
            input = input.to(device)
            if args.algorithm == 'asymmetric21':
                clipped_recon_image, rd_loss, distortion, bpp_feature, bpp_z, bpp = net(input, l=0, s1=0, s2=0)
            else:
                if args.mode == 'MSE':
                    clipped_recon_image, rd_loss, distortion, bpp_feature, bpp_z, bpp = net(input, args.train_lambda, use_ssim=False)
                else:
                    clipped_recon_image, rd_loss, distortion, bpp_feature, bpp_z, bpp = net(input, args.train_lambda, use_ssim=True)
            # for numerical stability
            if rd_loss.isnan().any() or rd_loss.isinf().any() or rd_loss > 10000:
                continue
            optimizer.zero_grad()
            rd_loss.backward()

            # clip_gradient
            torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            if (args.global_step % args.cal_step) == 0:
                if args.mode == 'MSE':
                    Distortion.update(distortion.item())
                    if distortion.item() > 0:
                        psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10))  # calculate psnr using mse
                        psnrs.update(psnr.item())
                    else:
                        psnrs.update(100)
                else:
                    psnrs.update(0)
                    msssim = distortion
                    msssimDB = -10 * (torch.log(1 - msssim) / np.log(10))
                    Distortion.update(msssimDB.item())
                elapsed.update(time.time() - start_time)
                losses.update(rd_loss.item())
                bpps.update(bpp.item())
                bpp_features.update(bpp_feature.item())
                bpp_zs.update(bpp_z.item())

            if (args.global_step % args.print_freq) == 0:     # 100
                process = args.global_step / args.tot_step * 100.0
                log = (' | '.join([
                    f'Step [{args.global_step}/{args.tot_step}={process:.2f}%]',
                    f'Epoch {epoch}',
                    f'Time {elapsed.val:.3f} ({elapsed.avg:.3f})',
                    f'Lr {args.cur_lr}',
                    f'Total Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'Bpp {bpps.val:.5f} ({bpps.avg:.5f})',
                    f'Bpp_feature {bpp_features.val:.5f} ({bpp_features.avg:.5f})',
                    f'Bpp_z {bpp_zs.val:.5f} ({bpp_zs.avg:.5f})',
                    f'Distortion {Distortion.val:.5f} ({Distortion.avg:.5f})',
                ]))
                loop.set_description(f'Epoch [{epoch}]')
                # loop.set_postfix(Total_Loss=losses.val, PSNR=psnrs.val, Bpp=bpps.val, MSE=mse_losses.val)
                logger.info(log)
        save_model_epoch(net, epoch, save_path)  # save the model per epoch


def train_imagenet(train_loader, args):
    net.train()
    elapsed, losses, psnrs, bpps, bpp_features, bpp_zs, Distortion = [AverageMeter() for _ in range(7)]
    args.tot_step = args.tot_epoch * (len(train_dataset) // (args.batch_size))  # calculate total steps
    loop = tqdm(train_loader, total=len(train_dataset) // args.batch_size)
    for epoch in range(args.tot_epoch):
        logger.info("Epoch {} begin".format(epoch))
        adjust_learning_rate(optimizer, args.global_step)    # update lr first

        for input, label in loop:
            start_time = time.time()
            args.global_step += 1
            input = input.to(device)
            loss, dist_acc_loss, distortion, bpp, bpp_feature, bpp_z = net(input, label)
            # for numerical stability
            if loss.isnan().any() or loss.isinf().any() or loss > 10000:
                continue
            optimizer.zero_grad()
            loss.backward()

            # clip_gradient
            torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            if (args.global_step % args.cal_step) == 0:
                if args.mode == 'MSE':
                    Distortion.update(distortion.item())
                    if distortion.item() > 0:
                        psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10))  # calculate psnr using mse
                        psnrs.update(psnr.item())
                    else:
                        psnrs.update(100)
                else:
                    psnrs.update(0)
                    msssim = distortion
                    msssimDB = -10 * (torch.log(1 - msssim) / np.log(10))
                    Distortion.update(msssimDB.item())
                elapsed.update(time.time() - start_time)
                losses.update(loss.item())
                bpps.update(bpp.item())
                bpp_features.update(bpp_feature.item())
                bpp_zs.update(bpp_z.item())

            if (args.global_step % args.print_freq) == 0:     # 100
                process = args.global_step / args.tot_step * 100.0
                log = (' | '.join([
                    f'Step [{args.global_step}/{args.tot_step}={process:.2f}%]',
                    f'Epoch {epoch}',
                    f'Time {elapsed.val:.3f} ({elapsed.avg:.3f})',
                    f'Lr {args.cur_lr}',
                    f'Total Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'Bpp {bpps.val:.5f} ({bpps.avg:.5f})',
                    f'Bpp_feature {bpp_features.val:.5f} ({bpp_features.avg:.5f})',
                    f'Bpp_z {bpp_zs.val:.5f} ({bpp_zs.avg:.5f})',
                    f'Distortion {Distortion.val:.5f} ({Distortion.avg:.5f})',
                ]))
                loop.set_description(f'Epoch [{epoch}]')
                logger.info(log)
        save_model_epoch(net, epoch, save_path)


if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda:0' if args.device=='cuda' and torch.cuda.is_available() else 'cpu')
    if args.seed is not None:
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    # Log settings
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)

    save_path = os.path.join('checkpoints', args.name)
    save_log_path = os.path.join(save_path, args.log_name + '.txt')
    if args.name != '':
        os.makedirs(save_path, exist_ok=True)
        filehandler = logging.FileHandler(save_log_path)   # log保存到./checkpoints/name
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    logger.info("image compression training")
    logger.info("\n".join([f"Algorithm:{args.algorithm}",
                           f"Training_mode:{args.mode}",
                           f"total_epochs:{args.tot_epoch}",
                           f"bacth_size:{args.batch_size}",
                           f"train_lambda:{args.train_lambda}",
                           f"lr: base:{args.base_lr}, decay:{args.lr_decay}, decay_interval:{args.decay_interval}",
                           ]))
    logger.info(open(args.config).read())
    parse_config(args)

    # Build model
    if args.algorithm == 'transformer21':    # need GT labels, different dataset
        model = Transformer21()
        if args.pretrain != '':
            logger.info("loading model:{}".format(args.pretrain))
            load_model_epoch(model, args.pretrain)
        net = model.to(device)

        # Train
        optimizer = optim.Adam(net.parameters(), lr=args.base_lr)
        train_data_dir = args.train
        train_dataset = Datasets_ImageNet(train_data_dir, args.image_size)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=2)
        train_imagenet(train_loader)

    else:
        if args.algorithm == 'hyperprior18':
            model = Hyperprior18(args.out_channel_N, args.out_channel_M)
        elif args.algorithm == 'autoregressive18':
            model = Autoregressive18(args.out_channel)
        elif args.algorithm == 'asymmetric21':
            model = Asymmetric21()
        else:
            model = Transformer21()

        if args.pretrain != '':
            logger.info("loading model:{}".format(args.pretrain))
            load_model_epoch(model, args.pretrain)
        net = model.to(device)

        # Train
        optimizer = optim.Adam(net.parameters(), lr=args.base_lr)
        train_data_dir = args.train
        train_dataset = Datasets(train_data_dir, args.image_size)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=2)
        # logger.info(summary(net, input_size=(3, 256, 256), batch_size=4))
        args.tot_step = args.tot_epoch * (len(train_dataset)//args.batch_size)
        train(train_loader, args)
