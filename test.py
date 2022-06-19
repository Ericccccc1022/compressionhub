import os
import argparse
from models import *
from torch.utils.data import DataLoader
from torchsummary import summary
import logging
import matplotlib.pyplot as plt
from utils import *


# Some default settings
gpu_num = 1
algorithm = 'asymmetric21'
logger = logging.getLogger("ImageCompression")  # Build logger
out_channel = 192
out_channel_N = 128  # 192
out_channel_M = 192  # 320
log_name = 'log_16_test'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
assert algorithm in ['hyperprior18', 'autoregressive18', 'asymmetric21', 'transformer21']

parser = argparse.ArgumentParser(description='Pytorch reimplement')
parser.add_argument('-n', '--name', default='asymmetric21_ssim_test', help='experiment name')
parser.add_argument('-p', '--pretrain', default='./checkpoints/asymmetric21_ssim_test/epoch_10.pth.tar', help='load pretrain model')
parser.add_argument('--seed', default=10, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--val', dest='val', default='./data/kodak/', help='the path of validation dataset')


def testKodak(test_loader, *args):
    with torch.no_grad():
        net.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            input = input.to(device)
            if algorithm == 'asymmetric21':
                l, s1, s2 = args
                clipped_recon_image, rd_loss, distortion, bpp_feature, bpp_z, bpp, _ = net(input, l=l, s1=s1, s2=s2, use_ssim=True)
            else:
                clipped_recon_image, rd_loss, distortion, bpp_feature, bpp_z, bpp = net(input, 0.0003)

            # Save the image
            org_img = input[0, :, :, :].cpu().numpy()
            recon_img = clipped_recon_image[0, :, :, :].cpu().numpy()  # [3,256,256]
            org_img = np.transpose(org_img, (1, 2, 0))
            recon_img = np.transpose(recon_img, (1, 2, 0))  # [256,256,3]
            plt.figure(0, figsize=(12,16))
            plt.subplot(1, 2, 1), plt.imshow(org_img), plt.title('Origin', fontsize=22), plt.axis('off')
            plt.subplot(1, 2, 2), plt.imshow(recon_img), plt.title('Reconstructed', fontsize=22), plt.axis('off')
            if algorithm == 'asymmetric21':
                plt.savefig(os.path.join(save_path, f'figures_test/{s1}_{s2}_{l}_img{batch_idx}.png'))
            else:
                plt.savefig(os.path.join(save_path, f'figures_test/{batch_idx}.png'))
            plt.close(0)

            distortion, bpp_feature, bpp_z, bpp = \
                torch.mean(distortion), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / distortion) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image, input, data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            logger.info("Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(bpp, psnr, msssim, msssimDB))
            cnt += 1


        logger.info("Test on Kodak dataset:")
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        logger.info("Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
        return sumBpp, sumMsssimDB



if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(seed=args.seed)
    # Log settings
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)

    save_path = os.path.join('checkpoints', args.name)
    save_log_path = os.path.join(save_path, log_name + '.txt')
    if args.name != '':
        os.makedirs(save_path, exist_ok=True)
        filehandler = logging.FileHandler(save_log_path)   # log保存到./checkpoints/name
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    logger.info("Image Compression Testing")

    # Build model
    if algorithm == 'hyperprior18':
        model = Hyperprior18(out_channel_N, out_channel_M)
    elif algorithm == 'autoregressive18':
        model = Autoregressive18(out_channel)
    elif algorithm == 'asymmetric21':
        model = Asymmetric21()
    else:
        model = Transformer21()

    if args.pretrain != '':
        logger.info("loading model:{}".format(args.pretrain))
        load_model_epoch(model, args.pretrain)
    net = model.to(device)
    # logger.info(summary(net, input_size=(3, 256, 256), batch_size=1))

    test_dataset = TestKodakDataset(data_dir=args.val)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=1)
    # 测试
    if algorithm == 'asymmetric21':
        bpp_ls, msssimDB_ls = [], []
        ## CVR
        # for s1 in range(5):
        #     s2 = s1 + 1
        #     for l in range(9,-1,-1):
        #         l *= 0.1    # l:[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        #         sumBpp, sumMsssimDB = testKodak(test_loader, l, s1, s2)
        #         bpp_ls.append(sumBpp)
        #         msssimDB_ls.append(sumMsssimDB)
        # plt.plot(bpp_ls, msssimDB_ls, 'ro')
        # plt.ylabel('MSSSIM-DB')
        # plt.xlabel('Bits Per Pixel(BPP)')
        # plt.title('Averaged MS-SSIM on 24 Kodak Images')
        # plt.grid()
        # plt.show()
        # plt.savefig('./results/asymmetric21.png')

        ## DVR
        for s1 in range(6):
            s2 = s1
            l = 1
            sumBpp, sumMsssimDB = testKodak(test_loader, l, s1, s2)
            bpp_ls.append(sumBpp)
            msssimDB_ls.append(sumMsssimDB)
        plt.plot(bpp_ls, msssimDB_ls, 'ro:')
        plt.ylabel('MSSSIM-DB')
        plt.xlabel('Bits Per Pixel(BPP)')
        plt.title('Averaged MS-SSIM on 24 Kodak Images')
        plt.grid()
        plt.show()
        plt.savefig('./results/asymmetric21.png')
    else:
        testKodak(test_loader)

