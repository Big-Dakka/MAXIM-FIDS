###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
FaceID network for AI85/AI86

Optionally quantize/clamp activations
"""
import torch.nn as nn

import ai8x


class AI85FDSNet(nn.Module):
    """
    Simple FaceNet Model
    """
    def __init__(
            self,
            num_classes=10,  # pylint: disable=unused-argument
            num_channels=3,
            dimensions=(128, 128),  # pylint: disable=unused-argument
            bias=True,
            **kwargs
    ):
        super().__init__()

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 16, 3, padding=1,
                                          bias=False, **kwargs)
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(16, 32, 3, pool_size=2, pool_stride=2,
                                                 padding=1, bias=False, **kwargs)
        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(32, 32, 3, pool_size=2, pool_stride=2,
                                                 padding=1, bias=bias, **kwargs)
        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(32, 64, 3, pool_size=2, pool_stride=2,
                                                 padding=1, bias=bias, **kwargs)
        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(64, 64, 3, pool_size=2, pool_stride=2,
                                                 padding=1, bias=bias, **kwargs)
        self.conv6 = ai8x.FusedConv2dReLU(64, 64, 3, padding=1, bias=bias, **kwargs)
        self.conv7 = ai8x.FusedConv2dReLU(64, 64, 3, padding=1, bias=bias, **kwargs)
        self.conv8 = ai8x.FusedMaxPoolConv2d(64, 512, 1, pool_size=2, pool_stride=2,
                                             padding=0, bias=False, **kwargs)
        self.conv9 = ai8x.Conv2d(512, num_classes, 1, stride=1, padding=0, wide=True, bias=bias, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = x.view(x.size(0), -1)
        return x


def ai85fdsnet(pretrained=False, **kwargs):
    """
    Constructs a FaceIDNet model.
    """
    assert not pretrained
    return AI85FDSNet(**kwargs)


models = [
    {
        'name': 'ai85fdsnet',
        'min_input': 1,
        'dim': 3,
    },
]
