import torch
import numpy as np
import torch.nn as nn
from .utils import load_url

__all__ = ['attention_u_net_deep_ds4x']

model_urls = {
    'attention_u_net': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/A-Unet-512x1024_encoder_epoch268_last.pth',
}

# def first_conv(in_channels, out_channels, step=1):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 5, stride=1,
#                   padding=1, groups=1, bias=False),
#         nn.LeakyReLU(0.01),
#         nn.InstanceNorm2d(out_channels, affine=True),
#         nn.Conv2d(out_channels, out_channels, 5, stride=step,
#                   padding=1, groups=1, bias=False),
#         nn.LeakyReLU(0.01),
#         nn.BatchNorm2d(out_channels, affine=True)
#     )


def double_conv(in_channels, out_channels, step=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=1,
                  padding=1, groups=1, bias=False),
        nn.LeakyReLU(0.01),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=1,
                  padding=1, groups=1, bias=False),
        nn.LeakyReLU(0.01),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=step,
                  padding=1, groups=1, bias=False),
        nn.LeakyReLU(0.01),
        nn.BatchNorm2d(out_channels, affine=True),
        nn.Conv2d(out_channels, out_channels, 3, stride=1,
                  padding=1, groups=1, bias=False),
        nn.LeakyReLU(0.01),
        nn.InstanceNorm2d(out_channels, affine=True)
    )


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(0.01)
        self.expand = nn.Conv2d(in_channels=in_channels // 8, out_channels=in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, x):
        output = self.norm(self.expand(self.relu(self.squeeze(self.pooling_layer(x)))))
        return output


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.merge_layer = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.01)
        self.norm = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, 1, True)
        max_out, __ = torch.max(x, 1, True)
        output = torch.cat([avg_out, max_out], dim=1)
        output = self.relu(self.merge_layer(output))
        output = self.norm(output)
        return output


class SimpleMixedAttention(nn.Module):
    def __init__(self, in_channels):
        super(SimpleMixedAttention, self).__init__()
        self.mixed_squeeze = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.01)
        self.mixed_expand = nn.Conv2d(in_channels=in_channels // 8, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, x):
        output = self.norm(self.mixed_expand(self.relu(self.mixed_squeeze(x))))
        return output


class AttentionUNet(nn.Module):
    # only three downsampling stages
    def __init__(self, in_ch, width, attention_mode):
        super(AttentionUNet, self).__init__()
        self.attention_type = attention_mode
        # if class_no > 2:
        #     self.final_in = class_no
        # else:
        #     self.final_in = 1
        self.w1 = width
        self.w2 = width*2
        self.w3 = width*4
        self.w4 = width*8
        # HRnet head
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.1) 
        # conv3 represent BOTTLENECK in HRnet
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        #
        self.dconv_down1 = double_conv(64, self.w1, step=1)
        self.dconv_down2 = double_conv(self.w1, self.w2, step=2)
        self.dconv_down3 = double_conv(self.w2 + self.w1, self.w3, step=2)
        self.dconv_down4 = double_conv(self.w3 + self.w2, self.w4, step=2)
        self.bridge = double_conv(self.w4 + self.w3, self.w4, step=1)
        self.bridge_2 = double_conv(self.w4, self.w4, step=1)
        self.bridge_3 = double_conv(self.w4, self.w4, step=1)
        self.bridge_4 = double_conv(self.w4, self.w4, step=1)
        self.bridge_5 = double_conv(self.w4, self.w4, step=1)
        self.bridge_6 = double_conv(self.w4, self.w4, step=1)
        self.bridge_7 = double_conv(self.w4, self.w4, step=1)
        self.bridge_8 = double_conv(self.w4, self.w4, step=1)
        self.bridge_9 = double_conv(self.w4, self.w4, step=1)
        self.bridge_10 = double_conv(self.w4, self.w4, step=1)

        self.dconv_up3 = double_conv(self.w3 + self.w4, self.w3, step=1)
        self.dconv_up2 = double_conv(self.w2 + self.w3, self.w2, step=1)
        self.dconv_up1 = double_conv(self.w1 + self.w2, self.w1, step=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.max_pool = nn.MaxPool2d(2)

        # self.conv_last = nn.Conv2d(width, self.final_in, 1)

        # Match resolution between global key and local query:
        self.s4_s3_match_res = nn.PixelShuffle(2)
        self.s4_s3_match_dim = nn.Conv2d(self.w4 // 4, self.w3, kernel_size=1, groups=1, bias=True)

        self.s4_s2_match_res = nn.PixelShuffle(4)
        self.s4_s2_match_dim = nn.Conv2d(self.w4 // 16, self.w2, kernel_size=1, groups=1, bias=True)

        self.s4_s1_match_res = nn.PixelShuffle(8)
        self.s4_s1_match_dim = nn.Conv2d(self.w4 // 64, self.w1, kernel_size=1, groups=1, bias=True)

        if self.attention_type == 'channel':
            self.s3_attention = ChannelAttention(self.w3)
            self.s2_attention = ChannelAttention(self.w2)
            self.s1_attention = ChannelAttention(self.w1)
        elif self.attention_type == 'spatial':
            self.s3_attention = SpatialAttention()
            self.s2_attention = SpatialAttention()
            self.s1_attention = SpatialAttention()
        elif self.attention_type == 'mixed':
            self.s3_attention = SimpleMixedAttention(self.w3)
            self.s2_attention = SimpleMixedAttention(self.w2)
            self.s1_attention = SimpleMixedAttention(self.w1)
        self.dconv_up3_attention = ChannelAttention(self.w3)
        self.dconv_up2_attention = ChannelAttention(self.w2)
        self.dconv_up1_attention = ChannelAttention(self.w1)
        self.dconv_up0 = double_conv(self.w1, self.w1, step=1)
    def forward(self, x, return_feature_maps=False):
        # HRnet head
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # Unet
        s1 = self.dconv_down1(x)
        s2 = self.dconv_down2(s1)
        s3 = self.dconv_down3(torch.cat([s2, self.max_pool(s1)], dim=1))
        s4 = self.dconv_down4(torch.cat([s3, self.max_pool(s2)], dim=1))
        s4 = self.bridge(torch.cat([s4, self.max_pool(s3)], dim=1))
        s4 = self.bridge_2(s4) + s4
        s4 = self.bridge_3(s4) + s4
        s4 = self.bridge_4(s4) + s4
        s4 = self.bridge_5(s4) + s4
        s4 = self.bridge_6(s4) + s4
        s4 = self.bridge_7(s4) + s4
        s4 = self.bridge_8(s4) + s4
        s4 = self.bridge_9(s4) + s4
        s4 = self.bridge_10(s4) + s4
        #
        global_s3 = self.s4_s3_match_dim(self.s4_s3_match_res(s4)) + s3
        global_s2 = self.s4_s2_match_dim(self.s4_s2_match_res(s4)) + s2
        global_s1 = self.s4_s1_match_dim(self.s4_s1_match_res(s4)) + s1
        #
        a_s3 = self.s3_attention(global_s3)*s3 + s3
        a_s2 = self.s2_attention(global_s2)*s2 + s2
        a_s1 = self.s1_attention(global_s1)*s1 + s1
        #
        output = torch.cat([a_s3, self.upsample(s4)], dim=1)
        output = self.dconv_up3(output)
        output = self.dconv_up3_attention(output) * output + output
        #
        output = torch.cat([a_s2, self.upsample(output)], dim=1)
        output = self.dconv_up2(output)
        output = self.dconv_up2_attention(output) * output + output
        #
        output = torch.cat([a_s1, self.upsample(output)], dim=1)
        output = self.dconv_up1(output)
        output = self.dconv_up1_attention(output) * output + output
        #
        output = self.dconv_up0(output)

        return [output]

def attention_u_net_deep_ds4x(pretrained=False, **kwargs):
    model = AttentionUNet(in_ch=3, width=48, attention_mode='mixed')
    # if pretrained:
        # model.load_state_dict(load_url(model_urls['attention_u_net']), strict=False)
    return model
