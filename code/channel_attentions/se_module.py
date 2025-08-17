import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块


class SELayer(nn.Module):  # 定义SE（Squeeze-and-Excitation）层类
    def __init__(self, channel, reduction=16):  # 初始化函数，参数：通道数、降维比例（默认16）
        super(SELayer, self).__init__()  # 调用父类初始化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应全局平均池化，输出1x1
        self.fc = nn.Sequential(  # 全连接序列模块
            nn.Linear(channel, channel // reduction, bias=False),  # 第一个线性层：降维到1/reduction
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(channel // reduction, channel, bias=False),  # 第二个线性层：恢复到原始通道数
            nn.Sigmoid()  # Sigmoid激活函数，生成0-1之间的注意力权重
        )

    def execute(self, x):  # 前向传播函数
        b, c, _, _ = x.size()  # 获取批次大小和通道数
        y = self.avg_pool(x).view(b, c)  # 全局平均池化并展平为(批次, 通道)
        y = self.fc(y).view(b, c, 1, 1)  # 通过全连接层生成注意力权重并重塑为(批次, 通道, 1, 1)
        return x * y.expand_as(x)  # 将注意力权重扩展并与输入特征相乘


def main():  # 主函数，用于测试
    attention_block = SELayer(64)  # 创建SE层，64个通道
    input = torch.rand([4, 64, 32, 32])  # 创建测试输入张量，形状为(4, 64, 32, 32)
    output = attention_block(input)  # 通过SE层进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
