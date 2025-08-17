# ResNest: Split-attention networks (arXiv 2020)
# ResNest：分割注意力网络 (arXiv 2020)
import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):  # 定义函数：使数值能被除数整除
    min_value = min_value or divisor  # 如果未指定最小值，则使用除数作为最小值
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)  # 计算新的可整除值
    # Make sure that round down does not go down by more than 10%.
    # 确保向下舍入不会下降超过10%
    if new_v < round_limit * v:  # 如果新值过小（小于原值的90%）
        new_v += divisor  # 增加一个除数
    return new_v  # 返回调整后的值


class RadixSoftmax(nn.Module):  # 定义基数Softmax类
    def __init__(self, radix, cardinality):  # 初始化函数，参数：基数、基数
        super(RadixSoftmax, self).__init__()  # 调用父类初始化
        self.radix = radix  # 存储基数（分支数）
        self.cardinality = cardinality  # 存储基数（组数）

    def execute(self, x):  # 前向传播函数
        batch = x.size(0)  # 获取批次大小
        if self.radix > 1:  # 如果基数大于1（多分支）
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)  # 重塑张量形状并转置维度
            x = nn.softmax(x, dim=1)  # 在基数维度上应用softmax
            x = x.reshape(batch, -1)  # 重新展平张量
        else:  # 如果基数等于1（单分支）
            x = x.sigmoid()  # 使用sigmoid激活
        return x  # 返回处理后的张量


class SplitAttn(nn.Module):  # 定义分割注意力类
    """Split-Attention (aka Splat)
    分割注意力（又称Splat）
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=None,  # 初始化函数，包含卷积参数
                 dilation=1, groups=1, bias=False, radix=2, rd_ratio=0.25, rd_channels=None, rd_divisor=8,  # 扩张、分组、偏置、基数、缩减比例等参数
                 act_layer=nn.ReLU, norm_layer=None, drop_block=None, **kwargs):  # 激活层、归一化层、dropout等参数
        super(SplitAttn, self).__init__()  # 调用父类初始化
        out_channels = out_channels or in_channels  # 如果未指定输出通道，则使用输入通道数
        self.radix = radix  # 存储基数（分支数）
        self.drop_block = drop_block  # 存储dropout模块
        mid_chs = out_channels * radix  # 计算中间通道数（输出通道数乘以基数）
        if rd_channels is None:  # 如果未指定注意力通道数
            attn_chs = make_divisible(  # 计算注意力通道数，确保能被除数整除
                in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor)
        else:  # 如果指定了注意力通道数
            attn_chs = rd_channels * radix  # 注意力通道数乘以基数

        padding = kernel_size // 2 if padding is None else padding  # 如果未指定填充，则使用卷积核大小的一半
        self.conv = nn.Conv2d(  # 主卷积层
            in_channels, mid_chs, kernel_size, stride, padding, dilation,  # 输入通道到中间通道的卷积
            groups=groups * radix, bias=bias, **kwargs)  # 分组数为原分组数乘以基数
        self.bn0 = norm_layer(mid_chs) if norm_layer else nn.Identity()  # 第一个批归一化层或恒等映射
        self.act0 = act_layer()  # 第一个激活层
        self.fc1 = nn.Conv2d(out_channels, attn_chs, 1, groups=groups)  # 第一个1x1卷积（用于注意力计算）
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()  # 第二个批归一化层或恒等映射
        self.act1 = act_layer()  # 第二个激活层
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)  # 第二个1x1卷积（生成注意力权重）
        self.rsoftmax = RadixSoftmax(radix, groups)  # 基数softmax模块

    def execute(self, x):  # 前向传播函数
        x = self.conv(x)  # 通过主卷积层
        x = self.bn0(x)  # 应用第一个批归一化
        if self.drop_block is not None:  # 如果有dropout模块
            x = self.drop_block(x)  # 应用dropout
        x = self.act0(x)  # 应用第一个激活函数

        B, RC, H, W = x.shape  # 获取张量形状：批次、通道、高、宽
        if self.radix > 1:  # 如果基数大于1（多分支）
            x = x.reshape((B, self.radix, RC // self.radix, H, W))  # 重塑为(B, radix, C, H, W)
            x_gap = x.sum(dim=1)  # 在基数维度上求和，得到全局特征
        else:  # 如果基数等于1（单分支）
            x_gap = x  # 直接使用原特征
        x_gap = x_gap.mean(2, keepdims=True).mean(3, keepdims=True)  # 全局平均池化，保持维度
        x_gap = self.fc1(x_gap)  # 通过第一个1x1卷积
        x_gap = self.bn1(x_gap)  # 应用第二个批归一化
        x_gap = self.act1(x_gap)  # 应用第二个激活函数
        x_attn = self.fc2(x_gap)  # 通过第二个1x1卷积，生成注意力权重

        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)  # 应用基数softmax并重塑为广播形状
        if self.radix > 1:  # 如果基数大于1（多分支）
            out = (x * x_attn.reshape((B, self.radix,  # 将注意力权重重塑并与特征相乘
                                       RC // self.radix, 1, 1))).sum(dim=1)  # 在基数维度上加权求和
        else:  # 如果基数等于1（单分支）
            out = x * x_attn  # 直接与注意力权重相乘
        return out  # 返回输出


def main():  # 主函数，用于测试
    attention_block = SplitAttn(64)  # 创建一个分割注意力模块，输入通道数为64
    input = torch.ones([4, 64, 32, 32])  # 创建测试输入张量，形状为(4, 64, 32, 32)
    output = attention_block(input)  # 通过分割注意力模块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
