from torch import nn

class MaskConv2d(nn.Conv2d):
    """
    Reimplementation of MaskConv in PixelCNN
    which is adopted in "Autoregressive" paper.
    This can be used in autoregressive-based networks.
    Args:
        mask_type: str, 'A' or 'B', corresponding to the central pixel be 0 or 1
    """
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ('A','B')
        self.register_buffer('mask', self.weight.data.clone())  # won't be updated
        _, _, H, W = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, H // 2, W // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, H // 2+1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

