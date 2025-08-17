import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块


class MHSA(nn.Module):  # 定义多头自注意力类（Multi-Head Self-Attention）
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):  # 初始化函数
        super(MHSA, self).__init__()  # 调用父类初始化
        self.num_heads = num_heads  # 注意力头的数量
        head_dim = dim // num_heads  # 每个头的维度
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子：1/sqrt(head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # QKV线性变换：一次性生成查询、键、值
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力权重的Dropout
        self.proj = nn.Linear(dim, dim)  # 输出投影层
        self.proj_drop = nn.Dropout(proj_drop)  # 输出的Dropout

    def execute(self, x):  # 前向传播函数
        b, n, c = x.shape  # 获取输入张量的形状：批次、序列长度、特征维度
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c //  # 生成QKV并重塑形状
                                  self.num_heads).permute(2, 0, 3, 1, 4)  # 调整维度顺序：(3, B, num_heads, N, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离查询、键、值张量

        # attn = nn.bmm(q,k.transpose(0,1,3,2))*self.scale
        attn = nn.bmm_transpose(q, k)*self.scale  # 计算注意力分数：Q * K^T * scale（注意：这里可能有API错误）

        attn = nn.softmax(attn, dim=-1)  # 对注意力分数应用softmax

        attn = self.attn_drop(attn)  # 对注意力权重应用dropout

        out = nn.bmm(attn, v)  # 注意力权重与值的矩阵乘法
        out = out.transpose(0, 2, 1, 3).reshape(b, n, c)  # 重塑输出形状并调整维度顺序
        out = self.proj(out)  # 通过输出投影层
        out = self.proj_drop(out)  # 应用输出dropout

        return out  # 返回最终输出


def main():  # 主函数，用于测试
    attention_block = MHSA(64)  # 创建多头自注意力模块，64维特征
    input = torch.rand([4, 128, 64])  # 创建随机输入张量：(批次=4, 序列长度=128, 特征维度=64)
    output = attention_block(input)  # 通过多头自注意力模块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
