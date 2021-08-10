import torch
import numpy as np
import torch.nn as nn

__all__ = ['u_net_nonsyn']

def double_conv(in_channels, out_channels, step=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=1,
                  padding=1, groups=1, bias=False),
        nn.ReLU6(inplace=True),
        nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False),
        nn.Conv2d(out_channels, out_channels, 3, stride=step,
                  padding=1, groups=1, bias=False),
        nn.ReLU6(inplace=True),
        nn.BatchNorm2d(out_channels, affine=True, track_running_stats=False)
    )


class UNet(nn.Module):
    def __init__(self, in_ch, width, class_no, res_mode):
        super().__init__()
        self.identity_add = res_mode
        # change class number here:
        if class_no > 2:
            self.final_in = class_no
        else:
            self.final_in = 1
        self.w1 = width
        self.w2 = width*2
        self.w3 = width*4
        self.w4 = width*8
        #
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        #
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv_last = nn.Conv2d(width, self.final_in, 1)
        #
        if self.identity_add is False:
            self.dconv_down1 = double_conv(in_ch, self.w1, step=1)
            self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
            self.dconv_down3 = double_conv(self.w2, self.w3, step=2)
            self.dconv_down4 = double_conv(self.w3, self.w4, step=2)
            self.bridge = double_conv(self.w4, self.w4, step=1)
            self.bridge_2 = double_conv(self.w4, self.w4, step=1)
            self.bridge_3 = double_conv(self.w4, self.w4, step=1)
            self.bridge_4 = double_conv(self.w4, self.w4, step=1)
        else:
            # identity mapping in the encoder:
            # when the dimensionality is the same, element-wise addition
            # when the dimensioanlity is not the same, concatenation
            self.dconv_down1 = double_conv(in_ch, self.w1, step=1)
            self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
            self.dconv_down3 = double_conv(self.w2+self.w1, self.w3, step=2)
            self.dconv_down4 = double_conv(self.w3+self.w2, self.w4, step=2)
            self.bridge = double_conv(self.w4+self.w3, self.w4, step=1)
            self.bridge_2 = double_conv(self.w4, self.w4, step=1)
            self.bridge_3 = double_conv(self.w4, self.w4, step=1)
            self.bridge_4 = double_conv(self.w4, self.w4, step=1)
            #
            self.max_pool = nn.MaxPool2d(2)

    def forward(self, x, return_feature_maps=False):
        if self.identity_add is False:
            conv1 = self.dconv_down1(x)
            conv2 = self.dconv_down2(conv1)
            conv3 = self.dconv_down3(conv2)
            conv4 = self.dconv_down4(conv3)
            conv4 = self.bridge_4(self.bridge_3(self.bridge_2(self.bridge(conv4))))

        else:
            conv1 = self.dconv_down1(x)
            conv2 = self.dconv_down2(conv1)
            conv3 = self.dconv_down3(torch.cat([conv2, self.max_pool(conv1)], dim=1))
            conv4 = self.dconv_down4(torch.cat([conv3, self.max_pool(conv2)], dim=1))
            conv4 = self.bridge(torch.cat([conv4, self.max_pool(conv3)], dim=1))
            conv4 = self.bridge_2(conv4) + conv4
            conv4 = self.bridge_3(conv4) + conv4
            conv4 = self.bridge_4(conv4) + conv4

        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        out = self.dconv_up1(x)
        # out = self.conv_last(x)

        return [out]

def u_net_nonsyn(pretrained=False, width=32, **kwargs):
    model = UNet(in_ch=3, width=width, class_no=7, res_mode=True)
    #if pretrained:
        # model.load_state_dict(load_url(model_urls['hrnetv2']), strict=False)
    return model
