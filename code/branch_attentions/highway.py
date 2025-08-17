import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块


class Highway(nn.Module):  # 定义Highway网络类
    def __init__(self, dim, num_layers=2):  # 初始化函数，参数：特征维度、网络层数（默认2层）

        super(Highway, self).__init__()  # 调用父类初始化

        self.num_layers = num_layers  # 存储网络层数

        self.nonlinear = nn.ModuleList(  # 创建非线性变换层的模块列表
            [nn.Linear(dim, dim) for _ in range(num_layers)])  # 每层都是从dim到dim的线性变换
        self.linear = nn.ModuleList([nn.Linear(dim, dim)  # 创建线性变换层的模块列表
                                    for _ in range(num_layers)])  # 每层都是从dim到dim的线性变换
        self.gate = nn.ModuleList([nn.Linear(dim, dim)  # 创建门控层的模块列表
                                  for _ in range(num_layers)])  # 每层都是从dim到dim的线性变换

        self.f = nn.ReLU()  # ReLU激活函数，用于非线性变换
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数，用于门控机制

    def execute(self, x):  # 前向传播函数
        """
            :param x: tensor with shape of [batch_size, size]  参数x：形状为[batch_size, size]的张量
            :return: tensor with shape of [batch_size, size]  返回：形状为[batch_size, size]的张量
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            应用 σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) 变换 | G和Q是仿射变换，
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            f是非线性变换，σ(x)是带sigmoid非线性的仿射变换
            and ⨀ is element-wise multiplication
            ⨀是逐元素乘法
            """

        for layer in range(self.num_layers):  # 遍历每一层
            gate = self.sigmoid(self.gate[layer](x))  # 计算门控值：通过线性变换后使用sigmoid激活
            nonlinear = self.f(self.nonlinear[layer](x))  # 计算非线性变换：通过线性变换后使用ReLU激活
            linear = self.linear[layer](x)  # 计算线性变换（直接的仿射变换）
            x = gate * nonlinear + (1 - gate) * linear  # Highway变换：门控值控制非线性和线性变换的混合
            print(x.size())  # 打印当前层输出的尺寸（调试用）
        return x  # 返回最终输出


def main():  # 主函数，用于测试
    attention_block = Highway(32)  # 创建一个Highway网络，特征维度为32
    input = torch.rand([4, 64, 32])  # 创建测试输入张量，形状为(4, 64, 32)
    output = attention_block(input)  # 通过Highway网络进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
