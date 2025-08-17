# Gated channel transformation for visual recognition (CVPR2020)
# 视觉识别的门控通道变换 (CVPR2020)
import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块


class GCT(nn.Module):  # 定义GCT（Gated Channel Transformation）类

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):  # 初始化函数
        super(GCT, self).__init__()  # 调用父类初始化

        self.alpha = torch.ones((1, num_channels, 1, 1))  # 初始化alpha参数，形状为(1, C, 1, 1)，初值为1
        self.gamma = torch.zeros((1, num_channels, 1, 1))  # 初始化gamma参数，形状为(1, C, 1, 1)，初值为0
        self.beta = torch.zeros((1, num_channels, 1, 1))  # 初始化beta参数，形状为(1, C, 1, 1)，初值为0
        self.epsilon = epsilon  # 存储epsilon值，用于数值稳定性
        self.mode = mode  # 存储模式（'l1'或'l2'）
        self.after_relu = after_relu  # 存储是否在ReLU之后应用的标志

    def execute(self, x):  # 前向传播函数

        if self.mode == 'l2':  # 如果使用L2模式
            embedding = (x.pow(2).sum(2, keepdims=True).sum(3, keepdims=True) +  # 计算L2嵌入：对每个通道在空间维度上计算平方和
                         self.epsilon).pow(0.5) * self.alpha  # 开平方根并乘以alpha参数
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdims=True) + self.epsilon).pow(0.5)  # 计算归一化系数：gamma除以嵌入的均方根

        elif self.mode == 'l1':  # 如果使用L1模式
            if not self.after_relu:  # 如果不是在ReLU之后
                _x = torch.abs(x)  # 对输入取绝对值
            else:  # 如果是在ReLU之后
                _x = x  # 直接使用输入（ReLU后已经非负）
            embedding = _x.sum(2, keepdims=True).sum(3, keepdims=True) * self.alpha  # 计算L1嵌入：对每个通道在空间维度上求和并乘以alpha参数
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdims=True) + self.epsilon)  # 计算归一化系数：gamma除以嵌入绝对值的均值
        else:  # 如果是未知模式
            print('Unknown mode!')  # 打印错误信息

        gate = 1. + torch.tanh(embedding * norm + self.beta)  # 计算门控值：1 + tanh(嵌入*归一化 + beta)

        return x * gate  # 返回输入乘以门控值


def main():  # 主函数，用于测试
    attention_block = GCT(64)  # 创建GCT模块，64个通道
    input = torch.rand([4, 64, 32, 32])  # 创建测试输入张量，形状为(4, 64, 32, 32)
    output = attention_block(input)  # 通过GCT模块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
