# PCT: Point Cloud Transformer (CVMJ 2021)
# PCT：点云变换器 (CVMJ 2021)

import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块


class SA_Layer(nn.Module):  # 定义自注意力层类（Self-Attention Layer）
    def __init__(self, channels):  # 初始化函数，参数：通道数
        super(SA_Layer, self).__init__()  # 调用父类初始化
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)  # 查询的1D卷积，降维到1/4
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)  # 键的1D卷积，降维到1/4
        self.q_conv.weight = self.k_conv.weight  # 查询和键共享权重
        self.v_conv = nn.Conv1d(channels, channels, 1)  # 值的1D卷积，保持原维度
        self.trans_conv = nn.Conv1d(channels, channels, 1)  # 变换卷积
        self.after_norm = nn.BatchNorm1d(channels)  # 批量归一化
        self.act = nn.ReLU()  # ReLU激活函数
        self.softmax = nn.Softmax(dim=-1)  # Softmax函数

    def execute(self, x):  # 前向传播函数
        x_q = self.q_conv(x).permute(0, 2, 1)  # 生成查询：(b, c, n) -> (b, n, c/4)
        x_k = self.k_conv(x)  # 生成键：(b, c/4, n)
        x_v = self.v_conv(x)  # 生成值：(b, c, n)
        energy = nn.bmm(x_q, x_k)  # 计算能量：查询与键的矩阵乘法，结果(b, n, n)
        attention = self.softmax(energy)  # 对能量应用softmax得到注意力权重
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # 归一化注意力权重，防止除零
        x_r = nn.bmm(x_v, attention)  # 注意力加权：值与注意力权重的矩阵乘法，结果(b, c, n)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))  # 残差变换：(x - x_r) -> 变换 -> 归一化 -> 激活
        x = x + x_r  # 最终残差连接：原输入加上变换后的结果
        return x  # 返回输出


def main():  # 主函数，用于测试
    attention_block = SA_Layer(64)  # 创建自注意力层，64通道
    input = torch.rand([4, 64, 32])  # 创建随机输入张量：(批次=4, 通道=64, 序列长度=32)
    output = attention_block(input)  # 通过自注意力层进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
