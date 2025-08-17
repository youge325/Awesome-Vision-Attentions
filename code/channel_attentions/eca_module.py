import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块


class ECALayer(nn.Module):  # 定义ECA（Efficient Channel Attention）层类
    """
    Constructs a ECA module.  构建ECA模块
    Args:  参数：
        k_size: Adaptive selection of kernel size  自适应选择卷积核大小
    """

    def __init__(self, k_size=3):  # 初始化函数，参数：卷积核大小（默认3）
        super(ECALayer, self).__init__()  # 调用父类初始化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，输出1x1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,  # 1D卷积层，输入输出都是1通道
                              padding=(k_size - 1) // 2, bias=False)  # 填充保持序列长度不变，无偏置
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def execute(self, x):  # 前向传播函数
        # feature descriptor on the global spatial information
        # 基于全局空间信息的特征描述符
        y = self.avg_pool(x)  # 全局平均池化，得到每个通道的全局特征 [B, C, 1, 1]

        # Two different branches of ECA module
        # ECA模块的两个不同分支
        y = self.conv(y.squeeze(-1).transpose(-1, -2)  # 移除最后一个维度并转置：[B, C, 1] -> [B, 1, C]
                      ).transpose(-1, -2).unsqueeze(-1)  # 1D卷积后再转置回来并增加维度：[B, 1, C] -> [B, C, 1, 1]

        y = self.sigmoid(y)  # 应用sigmoid激活，得到通道注意力权重

        return x * y.expand_as(x)  # 将注意力权重扩展并与输入特征相乘


def main():  # 主函数，用于测试
    attention_block = ECALayer()  # 创建ECA层，使用默认卷积核大小
    input = torch.rand([4, 64, 32, 32])  # 创建测试输入张量，形状为(4, 64, 32, 32)
    output = attention_block(input)  # 通过ECA层进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
