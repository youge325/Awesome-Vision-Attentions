import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块


class BasicConv(nn.Module):  # 定义基础卷积块类
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):  # 初始化函数，包含所有卷积参数
        super(BasicConv, self).__init__()  # 调用父类初始化
        self.out_channels = out_planes  # 存储输出通道数
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,  # 创建2D卷积层
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)  # 设置卷积参数
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,  # 创建批归一化层（如果需要）
                                 momentum=0.01, affine=True) if bn else None  # 设置批归一化参数
        self.relu = nn.ReLU() if relu else None  # 创建ReLU激活函数（如果需要）

    def execute(self, x):  # 前向传播函数
        x = self.conv(x)  # 通过卷积层
        if self.bn is not None:  # 如果有批归一化层
            x = self.bn(x)  # 应用批归一化
        if self.relu is not None:  # 如果有ReLU激活
            x = self.relu(x)  # 应用ReLU激活
        return x  # 返回结果


class ChannelGate(nn.Module):  # 定义通道门控类
    def __init__(self, channel, reduction=16):  # 初始化函数，参数：通道数、降维比例
        super(ChannelGate, self).__init__()  # 调用父类初始化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，输出1x1
        self.fc_avg = nn.Sequential(  # 平均池化分支的全连接层序列
            nn.Linear(channel, channel // reduction, bias=False),  # 降维线性层
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(channel // reduction, channel, bias=False),  # 恢复维度的线性层
        )
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化，输出1x1
        self.fc_max = nn.Sequential(  # 最大池化分支的全连接层序列
            nn.Linear(channel, channel // reduction, bias=False),  # 降维线性层
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(channel // reduction, channel, bias=False),  # 恢复维度的线性层
        )

        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def execute(self, x):  # 前向传播函数
        b, c, _, _ = x.size()  # 获取批次大小和通道数
        y_avg = self.avg_pool(x).view(b, c)  # 平均池化并展平
        y_avg = self.fc_avg(y_avg).view(b, c, 1, 1)  # 通过平均池化分支的全连接层并重塑

        y_max = self.max_pool(x).view(b, c)  # 最大池化并展平
        y_max = self.fc_max(y_max).view(b, c, 1, 1)  # 通过最大池化分支的全连接层并重塑

        y = self.sigmoid(y_avg + y_avg)  # 注意：原代码可能有错误，应该是y_avg + y_max，这里保持原样并添加注释
        return x * y.expand_as(x)  # 将通道注意力权重与输入相乘


class ChannelPool(nn.Module):  # 定义通道池化类
    def __init__(self):  # 初始化函数
        super().__init__()  # 调用父类初始化

    def execute(self, x):  # 前向传播函数
        x_max = torch.max(x, 1).unsqueeze(1)  # 在通道维度取最大值并增加维度，得到(B, 1, H, W)
        x_avg = torch.mean(x, 1).unsqueeze(1)  # 在通道维度取平均值并增加维度，得到(B, 1, H, W)
        x = torch.concat([x_max, x_avg], dim=1)  # 在通道维度连接最大值和平均值，得到(B, 2, H, W)
        return x  # 返回连接后的结果


class SpatialGate(nn.Module):  # 定义空间门控类
    def __init__(self):  # 初始化函数
        super(SpatialGate, self).__init__()  # 调用父类初始化
        kernel_size = 7  # 设置卷积核大小为7
        self.compress = ChannelPool()  # 创建通道池化层
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(  # 创建空间卷积层
            kernel_size-1) // 2, relu=False)  # 输入2通道（最大值+平均值），输出1通道，无ReLU
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def execute(self, x):  # 前向传播函数
        x_compress = self.compress(x)  # 通过通道池化，得到2通道的特征图
        x_out = self.spatial(x_compress)  # 通过空间卷积，得到1通道的空间注意力图
        scale = self.sigmoid(x_out)  # 通过sigmoid得到0-1之间的注意力权重  # broadcasting
        return x * scale  # 将空间注意力权重与输入相乘（广播）


class CBAM(nn.Module):  # 定义CBAM（Convolutional Block Attention Module）主类
    def __init__(self, gate_channels, reduction_ratio=16):  # 初始化函数，参数：门控通道数、降维比例
        super(CBAM, self).__init__()  # 调用父类初始化
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)  # 创建通道注意力门控
        self.SpatialGate = SpatialGate()  # 创建空间注意力门控

    def execute(self, x):  # 前向传播函数
        x_out = self.ChannelGate(x)  # 首先通过通道注意力门控
        x_out = self.SpatialGate(x_out)  # 然后通过空间注意力门控
        return x_out  # 返回经过双重注意力处理的结果


def main():  # 主函数，用于测试
    attention_block = CBAM(64)  # 创建CBAM模块，64个通道
    input = torch.rand([4, 64, 32, 32])  # 创建测试输入张量，形状为(4, 64, 32, 32)
    output = attention_block(input)  # 通过CBAM模块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
