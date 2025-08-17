# Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks (CVMJ2021)
# 超越自注意力：使用两个线性层的外部注意力机制用于视觉任务 (CVMJ2021)
import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块


class External_attention(nn.Module):  # 定义外部注意力类
    '''
    Arguments:  # 参数说明
        c (int): The input and output channel number.  # 输入和输出的通道数
    '''

    def __init__(self, c):  # 初始化函数，参数：通道数c
        super(External_attention, self).__init__()  # 调用父类初始化

        self.conv1 = nn.Conv2d(c, c, 1)  # 第一个1x1卷积层

        self.k = 64  # 外部记忆单元的数量
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)  # 第一个线性变换：c -> k

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)  # 第二个线性变换：k -> c
        self.linear_1.weight = self.linear_0.weight.permute(1, 0, 2)  # 权重共享：转置第一个线性层的权重

        self.conv2 = nn.Sequential(  # 第二个卷积序列
            nn.Conv2d(c, c, 1, bias=False),  # 1x1卷积
            nn.BatchNorm(c))  # 批量归一化（这里应该是nn.BatchNorm2d）

        self.relu = nn.ReLU()  # ReLU激活函数

    def execute(self, x):  # 前向传播函数
        idn = x  # 保存输入用于残差连接
        x = self.conv1(x)  # 通过第一个1x1卷积

        b, c, h, w = x.size()  # 获取张量的形状
        n = h*w  # 计算空间位置的总数
        x = x.view(b, c, h*w)   # 重塑为(b, c, n)形状

        attn = self.linear_0(x)  # 第一个线性变换：(b, c, n) -> (b, k, n)
        attn = nn.softmax(attn, dim=-1)  # 沿最后一维应用softmax得到注意力权重

        attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True))  # 归一化：防止除零，按k维度求和后归一化
        x = self.linear_1(attn)  # 第二个线性变换：(b, k, n) -> (b, c, n)

        x = x.view(b, c, h, w)  # 恢复到原始空间形状
        x = self.conv2(x)  # 通过第二个卷积序列
        x = x + idn  # 残差连接：加上原始输入
        x = self.relu(x)  # 应用ReLU激活
        return x  # 返回最终输出


def main():  # 主函数，用于测试
    attention_block = External_attention(64)  # 创建外部注意力模块，64通道
    input = torch.rand([4, 64, 32, 32])  # 创建随机输入张量
    output = attention_block(input)  # 通过外部注意力模块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
