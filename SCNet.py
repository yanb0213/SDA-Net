# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        # 第一个分支：1x1 卷积，用于提取局部特征，具有指定的扩张率
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        # 第二个分支：3x3 卷积，具有不同的扩张率，用于提取更大范围的局部特征
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=2 * rate, dilation=2 * rate, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

        # 第三个分支：全局平均池化分支，先进行 1x1 卷积
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        # self.branch5_bn = nn.BatchNorm2d(dim_out)
        self.branch5_relu = nn.ReLU(inplace=True)
        # 将三个分支的结果拼接后进行卷积操作，整合多尺度特征
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 3, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # 第一个分支的输出，提取局部特征
        conv1x1 = self.branch1(x)
        # 第二个分支的输出，提取更大范围的局部特征
        conv3x3_1 = self.branch2(x)

        # 全局平均池化，将特征图在空间维度上进行平均池化，得到全局特征
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        # 对全局特征进行 1x1 卷积操作
        global_feature = self.branch5_conv(global_feature)
        # 对全局特征进行归一化操作
        # global_feature = self.branch5_bn(global_feature)
        # 对全局特征进行激活操作
        global_feature = self.branch5_relu(global_feature)
        # 将全局特征上采样到原始尺寸，使用双线性插值
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # 将三个分支的特征在通道维度上拼接
        feature_cat = torch.cat([conv1x1, conv3x3_1, global_feature], dim=1)
        # 对拼接后的特征进行卷积操作，整合多尺度特征
        result = self.conv_cat(feature_cat)
        return result


class Resblock(nn.Module):
    """增强的残差块，支持下采样和上采样（修复通道匹配问题）"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_transpose=False):
        super().__init__()
        self.is_transpose = is_transpose
        self.in_channels = in_channels
        self.out_channels = out_channels

        if is_transpose:
            # 上采样残差块
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu2 = nn.ReLU()
        else:
            # 常规残差块
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu2 = nn.ReLU()

        # 残差连接适配
        if in_channels != out_channels or stride != 1:
            if is_transpose:
                self.shortcut = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride),
                    nn.BatchNorm2d(out_channels)
                )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = out + identity  # 避免隐式原地操作
        return out


class SCNet(nn.Module):
    """改进的CrossNet分割网络，编码器-解码器结构"""

    def __init__(self, in_channels=1, base_channels=64, num_encoder_blocks=4,
                 num_decoder_blocks=4, num_aea_blocks=2, num_heads=8, S=64,
                 out_channels=1):  # 新增out_channels参数，默认为1
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_encoder_blocks = num_encoder_blocks
        self.num_decoder_blocks = num_decoder_blocks
        self.out_channels = out_channels  # 新增：输出通道数

        # ===== 编码器路径 =====
        self.encoder = nn.ModuleList()

        # 初始卷积
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )

        # 编码器阶段
        for i in range(num_encoder_blocks):
            in_ch = base_channels * (2 ** i) if i == 0 else base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i + 1))

            stage = nn.Sequential(
                Resblock(in_ch, out_ch, stride=2),
                Resblock(out_ch, out_ch)
            )
            self.encoder.append(stage)

        # ===== 瓶颈层 =====et

        # 瓶颈层 ASPP
        bottleneck_ch = base_channels * (2 ** num_encoder_blocks)
        self.bottleneck = ASPP(dim_in=bottleneck_ch, dim_out=bottleneck_ch, rate=1)

        # ===== 解码器路径 =====
        self.decoder = nn.ModuleList()

        for i in range(num_decoder_blocks):
            # 编码器对应阶段的通道数（修正索引）
            encoder_ch = base_channels * (2 ** (num_encoder_blocks - 1 - i))
            # 解码器当前阶段的输出通道数
            out_ch = base_channels * (
                        2 ** (num_encoder_blocks - 1 - i)) if i < num_decoder_blocks - 1 else base_channels

            # 上采样残差块的输入通道数（编码器通道数 + 瓶颈通道数）
            upsample_in_ch = encoder_ch + (base_channels * (2 ** (num_encoder_blocks - i)))
            upsample_out_ch = encoder_ch

            stage = nn.Sequential(
                Resblock(upsample_in_ch, upsample_out_ch, is_transpose=True),
                Resblock(upsample_out_ch, out_ch),
                Resblock(out_ch, out_ch)
            )
            self.decoder.append(stage)

        # ===== 跳跃连接融合 =====
        self.skip_convs = nn.ModuleList()
        for i in range(num_encoder_blocks):
            in_ch = base_channels * (2 ** (i + 1))
            self.skip_convs.append(nn.Conv2d(in_ch, in_ch, 1))

        # 最终卷积层 - 修改输出通道数
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, self.out_channels, 1),  # 修改输出通道数
            nn.Sigmoid()
        )

    def forward(self, x):
        skips = []
        # 初始卷积
        x = self.init_conv(x)
        skips.append(x.clone())  # 确保保存的是副本

        # 编码器路径
        for i, stage in enumerate(self.encoder):
            x = stage(x)
            skips.append(x.clone())  # 确保保存的是副本

        # 瓶颈层
        x = self.bottleneck(x)

        # 解码器路径
        for i, stage in enumerate(self.decoder):
            # 获取对应编码器特征（从倒数第2个开始，与解码器阶段对应）
            if i < len(skips) - 1:
                skip = skips[-(i + 2)]
            else:
                skip = skips[0]  # 最后一层融合初始特征

            # 尺寸对齐（使用双线性插值）
            if skip.size()[2:] != x.size()[2:]:
                skip = F.interpolate(
                    skip,
                    size=x.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )

            # 特征拼接（修复通道数计算）
            x = torch.cat([x, skip], dim=1)
            x = stage(x)

        # 融合初始高分辨率特征
        if skips[0].size()[2:] != x.size()[2:]:
            skips[0] = F.interpolate(
                skips[0],
                size=x.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        x = torch.cat([x, skips[0]], dim=1)  # 通道拼接

        return self.final_conv(x)


if __name__ == "__main__":
    # 测试单通道输出
    input_tensor = torch.randn(2, 1, 120, 120, requires_grad=True)

    # 创建单通道模型
    model_single = CrossNet(in_channels=1, out_channels=1)
    output_single = model_single(input_tensor)

    # 创建十通道模型
    model_ten = CrossNet(in_channels=1, out_channels=10)
    output_ten = model_ten(input_tensor)

    print("输入尺寸:", input_tensor.shape)
    print("单通道输出尺寸:", output_single.shape)  # 应为 [2, 1, 256, 256]
    print("十通道输出尺寸:", output_ten.shape)  # 应为 [2, 10, 256, 256]
    print("单通道输出范围:", output_single.min().item(), "-", output_single.max().item())
    print("十通道输出范围:", output_ten.min().item(), "-", output_ten.max().item())
    print("梯度计算完成，模型可以正常训练")

