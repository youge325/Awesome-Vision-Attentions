# Is Attention Better Than Matrix Decomposition? (ICLR 2021)
# 注意力机制比矩阵分解更好吗？(ICLR 2021)
import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块
from contextlib import contextmanager  # 导入上下文管理器


@contextmanager  # 上下文管理器装饰器
def null_context():  # 定义空上下文函数
    yield  # 不做任何操作的上下文管理器


class NMF(nn.Module):  # 定义非负矩阵分解类
    def __init__(  # 初始化函数
        self,
        dim,  # 输入维度
        n,  # 空间位置数量
        ratio=8,  # 压缩比例，默认为8
        K=6,  # 迭代次数，默认为6
        eps=2e-8  # 数值稳定性参数
    ):
        super().__init__()  # 调用父类初始化
        r = dim // ratio  # 计算压缩维度

        self.D = torch.zeros((dim, r)).uniform_(0, 1)  # 初始化字典矩阵D，均匀分布[0,1]
        self.C = torch.zeros((r, n)).uniform_(0, 1)  # 初始化系数矩阵C，均匀分布[0,1]

        self.K = K  # 存储迭代次数

        self.eps = eps  # 存储数值稳定性参数

    def execute(self, x):  # 前向传播函数
        b, D, C, eps = x.shape[0], self.D, self.C, self.eps  # 获取批次大小和相关矩阵

        # x is made non-negative with relu as proposed in paper
        x = nn.relu(x)  # 使用ReLU确保输入非负，如论文所建议
        D = D.unsqueeze(0).repeat(b, 1, 1)  # 扩展D矩阵到批次维度
        C = C.unsqueeze(0).repeat(b, 1, 1)  # 扩展C矩阵到批次维度

        # transpose
        def t(tensor): return tensor.transpose(1, 2)  # 定义转置函数

        for k in reversed(range(self.K)):  # 反向迭代K次
            # only calculate gradients on the last step, per propose 'One-step Gradient'
            context = null_context if k == 0 else torch.no_grad  # 只在最后一步计算梯度（一步梯度）
            with context():  # 在上下文中执行
                C_new = C * ((t(D) @ x) / ((t(D) @ D @ C) + eps))  # 更新C矩阵：乘法更新规则
                D_new = D * ((x @ t(C)) / ((D @ C @ t(C)) + eps))  # 更新D矩阵：乘法更新规则
                C, D = C_new, D_new  # 更新C和D矩阵

        return D @ C  # 返回分解结果：D乘以C


class Hamburger(nn.Module):  # 定义汉堡包模块类
    def __init__(  # 初始化函数
        self,
        dim,  # 输入通道维度
        n,  # 空间位置数量
        inner_dim,  # 内部维度
        ratio=8,  # NMF压缩比例
        K=6  # NMF迭代次数
    ):
        super().__init__()  # 调用父类初始化

        self.lower_bread = nn.Conv1d(dim, inner_dim, 1, bias=False)  # 下层"面包"：降维1D卷积
        self.ham = NMF(inner_dim, n, ratio=ratio, K=K)  # "肉饼"：NMF模块
        self.upper_bread = nn.Conv1d(inner_dim, dim, 1, bias=False)  # 上层"面包"：升维1D卷积

    def execute(self, x):  # 前向传播函数
        input = x  # 保存原始输入用于残差连接
        shape = x.shape  # 保存原始形状
        x = x.flatten(2)  # 将空间维度展平：(B,C,H,W) -> (B,C,H*W)

        x = self.lower_bread(x)  # 通过下层面包：降维
        x = self.ham(x)  # 通过汉堡肉饼：NMF分解
        x = self.upper_bread(x)  # 通过上层面包：升维
        return input + x.reshape(shape)  # 残差连接：原输入加上重塑后的输出


def main():  # 主函数，用于测试
    attention_block = Hamburger(64, 32*32, 64, 8, 6)  # 创建汉堡包模块：64维，1024位置，内部64维，比例8，迭代6次
    input = torch.rand([4, 64, 32, 32])  # 创建随机输入张量
    output = attention_block(input)  # 通过汉堡包模块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
