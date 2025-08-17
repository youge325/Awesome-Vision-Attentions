# Bam: Bottleneck attention module(BMVC 2018)
# BAM：瓶颈注意力模块 (BMVC 2018)
import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块


class Flatten(nn.Module):  # 定义展平层类
    def execute(self, x):  # 前向传播函数
        return x.view(x.size(0), -1)  # 将输入张量展平为(batch_size, -1)


class ChannelGate(nn.Module):  # 定义通道门控类
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):  # 初始化函数，参数：门控通道数、降维比例、层数
        super(ChannelGate, self).__init__()  # 调用父类初始化
        self.gate_c = nn.ModuleList()  # 创建模块列表存储门控网络层
        self.gate_c.append(Flatten())  # 添加展平层
        gate_channels = [gate_channel]  # 初始化通道数列表，起始为输入通道数
        gate_channels += [gate_channel // reduction_ratio] * num_layers  # 添加中间层的降维通道数
        gate_channels += [gate_channel]  # 添加输出层通道数（恢复到原始通道数）
        for i in range(len(gate_channels) - 2):  # 遍历构建中间层
            self.gate_c.append(nn.Linear(  # 添加线性层
                gate_channels[i], gate_channels[i+1]))  # 从当前通道数到下一层通道数
            self.gate_c.append(nn.BatchNorm1d(gate_channels[i+1]))  # 添加1D批归一化层
            self.gate_c.append(nn.ReLU())  # 添加ReLU激活函数
        self.gate_c.append(nn.Linear(  # 添加最后一个线性层
            gate_channels[-2], gate_channels[-1]))  # 从倒数第二层到最后一层

    def execute(self, x):  # 前向传播函数
        avg_pool = nn.avg_pool2d(  # 全局平均池化
            x, x.size(2), stride=x.size(2))  # 池化核大小和步长都等于特征图尺寸，得到1x1输出
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(x)  # 通过门控网络，然后增加空间维度并扩展为输入尺寸


class SpatialGate(nn.Module):  # 定义空间门控类
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):  # 初始化函数
        super(SpatialGate, self).__init__()  # 调用父类初始化
        self.gate_s = nn.ModuleList()  # 创建模块列表存储空间门控网络层
        self.gate_s.append(nn.Conv2d(  # 添加第一个卷积层（降维）
            gate_channel, gate_channel//reduction_ratio, kernel_size=1))  # 1x1卷积进行通道降维
        self.gate_s.append(nn.BatchNorm2d(  # 添加2D批归一化层
            gate_channel//reduction_ratio))
        self.gate_s.append(nn.ReLU())  # 添加ReLU激活函数
        for i in range(dilation_conv_num):  # 循环添加膨胀卷积层
            self.gate_s.append(nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3,  # 3x3膨胀卷积
                                         padding=dilation_val, dilation=dilation_val))  # 填充和膨胀系数相等，保持特征图尺寸
            self.gate_s.append(nn.BatchNorm2d(  # 添加批归一化层
                gate_channel//reduction_ratio))
            self.gate_s.append(nn.ReLU())  # 添加ReLU激活函数
        self.gate_s.append(nn.Conv2d(  # 添加最后一个卷积层
            gate_channel//reduction_ratio, 1, kernel_size=1))  # 1x1卷积输出单通道的空间注意力图

    def execute(self, x):  # 前向传播函数
        return self.gate_s(x).expand_as(x)  # 通过空间门控网络，然后扩展为输入尺寸


class BAM(nn.Module):  # 定义BAM（Bottleneck Attention Module）主类
    def __init__(self, gate_channel):  # 初始化函数，参数：门控通道数
        super(BAM, self).__init__()  # 调用父类初始化
        self.channel_att = ChannelGate(gate_channel)  # 创建通道注意力门控
        self.spatial_att = SpatialGate(gate_channel)  # 创建空间注意力门控
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def execute(self, x):  # 前向传播函数
        att = 1 + self.sigmoid(self.channel_att(x) * self.spatial_att(x))  # 计算注意力：1 + sigmoid(通道注意力 * 空间注意力)
        return att * x  # 将注意力权重与输入特征相乘


def main():  # 主函数，用于测试
    attention_block = BAM(64)  # 创建BAM模块，64个通道
    input = torch.rand([4, 64, 32, 32])  # 创建测试输入张量，形状为(4, 64, 32, 32)
    output = attention_block(input)  # 通过BAM模块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
