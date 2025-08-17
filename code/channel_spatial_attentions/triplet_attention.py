# Rotate to attend: Convolutional triplet attention module (WACV 2021)
# 旋转注意：卷积三元组注意力模块 (WACV 2021)
import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块


class BasicConv(nn.Module):  # 基础卷积模块类
    def __init__(  # 初始化函数
        self,
        in_planes,  # 输入通道数
        out_planes,  # 输出通道数
        kernel_size,  # 卷积核大小
        stride=1,  # 步长，默认为1
        padding=0,  # 填充，默认为0
        dilation=1,  # 膨胀率，默认为1
        groups=1,  # 分组数，默认为1
        relu=True,  # 是否使用ReLU激活，默认为True
        bn=True,  # 是否使用批量归一化，默认为True
        bias=False,  # 是否使用偏置，默认为False
    ):
        super(BasicConv, self).__init__()  # 调用父类初始化
        self.out_channels = out_planes  # 存储输出通道数
        self.conv = nn.Conv2d(  # 定义二维卷积层
            in_planes,  # 输入通道数
            out_planes,  # 输出通道数
            kernel_size=kernel_size,  # 卷积核大小
            stride=stride,  # 步长
            padding=padding,  # 填充
            dilation=dilation,  # 膨胀率
            groups=groups,  # 分组数
            bias=bias,  # 偏置
        )
        self.bn = (  # 定义批量归一化层
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)  # 批量归一化参数
            if bn  # 如果启用批量归一化
            else None  # 否则为None
        )
        self.relu = nn.ReLU() if relu else None  # 如果启用ReLU则创建ReLU层，否则为None

    def execute(self, x):  # 前向传播函数
        x = self.conv(x)  # 卷积操作
        if self.bn is not None:  # 如果有批量归一化层
            x = self.bn(x)  # 应用批量归一化
        if self.relu is not None:  # 如果有ReLU激活函数
            x = self.relu(x)  # 应用ReLU激活
        return x  # 返回结果


class ZPool(nn.Module):  # Z池化模块类
    def execute(self, x):  # 前向传播函数
        return torch.concat(  # 沿通道维度连接
            (x.max(1).unsqueeze(1), x.mean(1).unsqueeze(1)), dim=1  # 最大值和均值池化结果
        )


class AttentionGate(nn.Module):  # 注意力门控模块类
    def __init__(self):  # 初始化函数
        super(AttentionGate, self).__init__()  # 调用父类初始化
        kernel_size = 7  # 卷积核大小为7
        self.compress = ZPool()  # Z池化压缩模块
        self.conv = BasicConv(  # 基础卷积层
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False  # 2通道到1通道，7x7卷积，padding=3，不使用ReLU
        )

    def execute(self, x):  # 前向传播函数
        x_compress = self.compress(x)  # 通过Z池化压缩特征
        x_out = self.conv(x_compress)  # 通过卷积层处理压缩后的特征
        scale = x_out.sigmoid()  # 使用sigmoid得到注意力权重
        return x * scale  # 原始输入乘以注意力权重


class TripletAttention(nn.Module):  # 三元组注意力模块类
    def __init__(self, no_spatial=False):  # 初始化函数，no_spatial控制是否包含空间注意力
        super(TripletAttention, self).__init__()  # 调用父类初始化
        self.cw = AttentionGate()  # 通道-宽度注意力门控
        self.hc = AttentionGate()  # 高度-通道注意力门控
        self.no_spatial = no_spatial  # 存储是否不使用空间注意力的标志
        if not no_spatial:  # 如果使用空间注意力
            self.hw = AttentionGate()  # 高度-宽度注意力门控

    def execute(self, x):  # 前向传播函数
        x_perm1 = x.permute(0, 2, 1, 3)  # 第一次维度置换：(B,C,H,W) -> (B,H,C,W)
        x_out1 = self.cw(x_perm1)  # 通过通道-宽度注意力
        x_out11 = x_out1.permute(0, 2, 1, 3)  # 置换回原始维度：(B,H,C,W) -> (B,C,H,W)
        x_perm2 = x.permute(0, 3, 2, 1)  # 第二次维度置换：(B,C,H,W) -> (B,W,H,C)
        x_out2 = self.hc(x_perm2)  # 通过高度-通道注意力
        x_out21 = x_out2.permute(0, 3, 2, 1)  # 置换回原始维度：(B,W,H,C) -> (B,C,H,W)
        if not self.no_spatial:  # 如果使用空间注意力
            x_out = self.hw(x)  # 通过高度-宽度注意力
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)  # 三个分支的平均：空间+通道-宽度+高度-通道
        else:  # 如果不使用空间注意力
            x_out = 1 / 2 * (x_out11 + x_out21)  # 两个分支的平均：通道-宽度+高度-通道
        return x_out  # 返回最终输出


def main():  # 主函数，用于测试
    attention_block = TripletAttention()  # 创建三元组注意力模块，使用默认参数
    input = torch.ones([4, 64, 32, 32])  # 创建测试输入张量，形状为(4, 64, 32, 32)
    output = attention_block(input)  # 通过三元组注意力模块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
