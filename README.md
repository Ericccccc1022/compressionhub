# PyTorch Reimplementation of Learned-Image-Compression

This repo is implementation for some learned-image-compression methods in **pytorch.**

## Install

The latest codes are tested on Ubuntu16.04LTS, CUDA11.0, PyTorch1.7 and Python 3.7

```shell
pip install -r requirements.txt
```

## Compression

### Data Preparation

We first use a subset of `Flicker dataset` to train the model, which contains `20745`images. But it turned out such data volume is not enough for the model to converge.

So we then turn to `COCO2017 dataset` which contains 12k images.

Following most of the settings, we use `Kodak24` as our test dataset, it can be downloaded by running:

```shell
bash ./data/download_kodak.sh
```

### Train

The coding style is follows the **"configuration" flow**. So we just need to change the config files to attain different models.

For example, for low bitrate in "hyperprior", the `out_channel_N=128` and the `out_channel_M=192`. And there is an example in `./configs/hyperprior_ssim_16.json`

To train the model:

```shell
python train.py --config ./configs/hyperprior_ssim_16.json --algorithm hyperprior18 --name hyperprior18_16_100e
```

### Test

```shell
python test.py --name hyperprior18_16_100e --pretrain ./checkpoints/hyperprior18_16_100e/epoch_70.pth.tar
```

### Performance

The results of `hyperprior18_16`：

```python
[test.py][L104][INFO] Image Compression Testing
[test.py][L115][INFO] loading model:./checkpoints/hyperprior18_16_100e/epoch_65.pth.tar
[test.py][L73][INFO] Bpp:0.429937, PSNR:24.380186, MS-SSIM:0.969602, MS-SSIM-DB:15.171616
[test.py][L73][INFO] Bpp:0.353845, PSNR:29.215752, MS-SSIM:0.968118, MS-SSIM-DB:14.964513
[test.py][L73][INFO] Bpp:0.278039, PSNR:30.977474, MS-SSIM:0.982311, MS-SSIM-DB:17.523043
[test.py][L73][INFO] Bpp:0.329194, PSNR:29.131145, MS-SSIM:0.973011, MS-SSIM-DB:15.688201
[test.py][L73][INFO] Bpp:0.447510, PSNR:24.964352, MS-SSIM:0.973069, MS-SSIM-DB:15.697458
[test.py][L73][INFO] Bpp:0.386457, PSNR:26.129536, MS-SSIM:0.971200, MS-SSIM-DB:15.406136
[test.py][L73][INFO] Bpp:0.306836, PSNR:29.646324, MS-SSIM:0.986440, MS-SSIM-DB:18.677479
[test.py][L73][INFO] Bpp:0.416949, PSNR:22.836014, MS-SSIM:0.975700, MS-SSIM-DB:16.143898
[test.py][L73][INFO] Bpp:0.270019, PSNR:29.535843, MS-SSIM:0.983062, MS-SSIM-DB:17.711279
[test.py][L73][INFO] Bpp:0.288293, PSNR:28.594925, MS-SSIM:0.981848, MS-SSIM-DB:17.410669
[test.py][L73][INFO] Bpp:0.359632, PSNR:27.419832, MS-SSIM:0.974023, MS-SSIM-DB:15.854036
[test.py][L73][INFO] Bpp:0.300656, PSNR:28.150160, MS-SSIM:0.976999, MS-SSIM-DB:16.382462
[test.py][L73][INFO] Bpp:0.481748, PSNR:22.258312, MS-SSIM:0.956534, MS-SSIM-DB:13.618530
[test.py][L73][INFO] Bpp:0.417155, PSNR:27.147121, MS-SSIM:0.971254, MS-SSIM-DB:15.414277
[test.py][L73][INFO] Bpp:0.303994, PSNR:27.364368, MS-SSIM:0.976436, MS-SSIM-DB:16.277424
[test.py][L73][INFO] Bpp:0.313961, PSNR:29.513273, MS-SSIM:0.976146, MS-SSIM-DB:16.224350
[test.py][L73][INFO] Bpp:0.310162, PSNR:29.942520, MS-SSIM:0.981909, MS-SSIM-DB:17.425253
[test.py][L73][INFO] Bpp:0.411444, PSNR:26.388832, MS-SSIM:0.967697, MS-SSIM-DB:14.907534
[test.py][L73][INFO] Bpp:0.334530, PSNR:27.811775, MS-SSIM:0.974569, MS-SSIM-DB:15.946379
[test.py][L73][INFO] Bpp:0.323449, PSNR:28.484882, MS-SSIM:0.982200, MS-SSIM-DB:17.495689
[test.py][L73][INFO] Bpp:0.321556, PSNR:26.926403, MS-SSIM:0.977092, MS-SSIM-DB:16.400145
[test.py][L73][INFO] Bpp:0.380560, PSNR:28.660519, MS-SSIM:0.966879, MS-SSIM-DB:14.798981
[test.py][L73][INFO] Bpp:0.254632, PSNR:31.240612, MS-SSIM:0.984432, MS-SSIM-DB:18.077583
[test.py][L73][INFO] Bpp:0.390071, PSNR:25.345428, MS-SSIM:0.973653, MS-SSIM-DB:15.792690
[test.py][L76][INFO] Test on Kodak dataset:
[test.py][L81][INFO] Dataset Average result---Bpp:0.350443, PSNR:27.586065, MS-SSIM:0.975174, MS-SSIM-DB:16.208735

```

![image-20220619150109358](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220619150109358.png)

