# Coordinate attention for efficient mobile network design (CVPR 2021)
# 高效移动网络设计的坐标注意力 (CVPR 2021)
import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块


class h_sigmoid(nn.Module):  # 定义硬sigmoid激活函数类
    def __init__(self):  # 初始化函数
        super(h_sigmoid, self).__init__()  # 调用父类初始化
        self.relu = nn.ReLU6()  # ReLU6激活函数，输出范围[0, 6]

    def execute(self, x):  # 前向传播函数
        return self.relu(x + 3) / 6  # 硬sigmoid：ReLU6(x + 3) / 6，输出范围[0, 1]


class h_swish(nn.Module):  # 定义硬Swish激活函数类
    def __init__(self):  # 初始化函数
        super(h_swish, self).__init__()  # 调用父类初始化
        self.sigmoid = h_sigmoid()  # 创建硬sigmoid实例

    def execute(self, x):  # 前向传播函数
        return x * self.sigmoid(x)  # 硬Swish：x * 硬sigmoid(x)


class CoordAtt(nn.Module):  # 定义坐标注意力模块类
    def __init__(self, inp, oup, reduction=32):  # 初始化函数，参数：输入通道数、输出通道数、降维比例
        super(CoordAtt, self).__init__()  # 调用父类初始化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 高度方向自适应平均池化，宽度压缩为1
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 宽度方向自适应平均池化，高度压缩为1

        mip = max(8, inp // reduction)  # 计算中间层通道数，最小为8

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)  # 1x1卷积降维
        self.bn1 = nn.BatchNorm2d(mip)  # 批归一化
        self.act = h_swish()  # 硬Swish激活函数

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)  # 生成高度注意力的1x1卷积
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)  # 生成宽度注意力的1x1卷积

    def execute(self, x):  # 前向传播函数
        identity = x  # 保存输入用于残差连接

        n, c, h, w = x.size()  # 获取输入张量的形状
        x_h = self.pool_h(x)  # 在高度方向池化，得到(n, c, h, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 在宽度方向池化并转置，得到(n, c, w, 1)

        y = torch.concat([x_h, x_w], dim=2)  # 在高度维度连接，得到(n, c, h+w, 1)
        y = self.conv1(y)  # 通过1x1卷积降维
        y = self.bn1(y)  # 批归一化
        y = self.act(y)  # 硬Swish激活

        x_h, x_w = torch.split(y, [h, w], dim=2)  # 分离高度和宽度特征
        x_w = x_w.permute(0, 1, 3, 2)  # 将宽度特征转置回正确的形状

        a_h = self.conv_h(x_h).sigmoid()  # 生成高度注意力权重并sigmoid激活
        a_w = self.conv_w(x_w).sigmoid()  # 生成宽度注意力权重并sigmoid激活

        out = identity * a_w * a_h  # 原始输入乘以宽度注意力再乘以高度注意力

        return out  # 返回坐标注意力处理后的结果


def main():  # 主函数，用于测试
    attention_block = CoordAtt(64, 64)  # 创建坐标注意力模块，输入输出都是64通道
    input = torch.rand([4, 64, 32, 32])  # 创建测试输入张量，形状为(4, 64, 32, 32)
    output = attention_block(input)  # 通过坐标注意力模块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
