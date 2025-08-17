# Simam: A simple, parameter-free attention module for convolutional neural networks (ICML 2021)
# SimAM：卷积神经网络的简单无参数注意力模块 (ICML 2021)
import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块


class simam_module(nn.Module):  # 定义SimAM模块类
    def __init__(self, e_lambda=1e-4):  # 初始化函数，参数：正则化参数lambda
        super(simam_module, self).__init__()  # 调用父类初始化

        self.activaton = nn.Sigmoid()  # Sigmoid激活函数
        self.e_lambda = e_lambda  # 存储正则化参数，用于数值稳定性

    def execute(self, x):  # 前向传播函数

        b, c, h, w = x.size()  # 获取输入张量的形状

        n = w * h - 1  # 计算空间维度的元素个数减1，用于方差计算

        x_minus_mu_square = (  # 计算每个像素与通道均值的平方差
            x - x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True)).pow(2)  # (x - μ)²
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=2, keepdims=True).sum(dim=3, keepdims=True) / n + self.e_lambda)) + 0.5  # 计算能量函数：分母为4*(方差+λ)，最后加0.5

        return x * self.activaton(y)  # 原始输入乘以sigmoid激活后的注意力权重


def main():  # 主函数，用于测试
    attention_block = simam_module()  # 创建SimAM模块，使用默认参数
    input = torch.ones([4, 64, 32, 32])  # 创建测试输入张量，形状为(4, 64, 32, 32)
    output = attention_block(input)  # 通过SimAM模块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
