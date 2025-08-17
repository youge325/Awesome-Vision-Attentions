# Improving convolutional networks with self-calibrated convolutions (CVPR 2020)
# 通过自校准卷积改进卷积网络 (CVPR 2020)
import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块


class SCConv(nn.Module):  # 定义自校准卷积类
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):  # 初始化函数
        super(SCConv, self).__init__()  # 调用父类初始化
        self.k2 = nn.Sequential(  # K2分支：下采样+卷积+归一化
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),  # 平均池化下采样，降低分辨率
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,  # 3x3卷积，步长为1
                      padding=padding, dilation=dilation,  # 设置填充和膨胀系数
                      groups=groups, bias=False),  # 分组卷积，无偏置
            norm_layer(planes),  # 批归一化层
        )
        self.k3 = nn.Sequential(  # K3分支：标准卷积+归一化
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,  # 3x3卷积，步长为1
                      padding=padding, dilation=dilation,  # 设置填充和膨胀系数
                      groups=groups, bias=False),  # 分组卷积，无偏置
            norm_layer(planes),  # 批归一化层
        )
        self.k4 = nn.Sequential(  # K4分支：标准卷积+归一化
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,  # 3x3卷积，使用指定步长
                      padding=padding, dilation=dilation,  # 设置填充和膨胀系数
                      groups=groups, bias=False),  # 分组卷积，无偏置
            norm_layer(planes),  # 批归一化层
        )

    def execute(self, x):  # 前向传播函数
        identity = x  # 保存输入作为恒等映射

        out = torch.sigmoid(torch.add(identity, nn.interpolate(  # 计算注意力权重
            self.k2(x), identity.size()[2:])))  # sigmoid(identity + 上采样的k2)，将k2的结果插值回原尺寸
        out = torch.multiply(self.k3(x), out)  # K3的输出与注意力权重相乘：k3 * sigmoid(identity + k2)
        out = self.k4(out)  # 通过K4分支进行最终的卷积处理

        return out  # 返回自校准卷积的结果


def main():  # 主函数，用于测试
    attention_block = SCConv(64, 64, stride=1,  # 创建自校准卷积模块
                             padding=2, dilation=2, groups=1, pooling_r=4, norm_layer=nn.BatchNorm2d)  # 设置各种参数
    input = torch.rand([4, 64, 32, 32])  # 创建测试输入张量，形状为(4, 64, 32, 32)
    output = attention_block(input)  # 通过自校准卷积模块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
