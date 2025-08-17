# SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
# SegFormer：基于变换器的语义分割的简单高效设计
import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块


class EfficientAttention(nn.Module):  # 定义高效注意力类
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):  # 初始化函数
        super().__init__()  # 调用父类初始化
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."  # 断言：维度必须被头数整除

        self.dim = dim  # 输入维度
        self.num_heads = num_heads  # 注意力头数
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子

        self.q = nn.Linear(dim, dim, bias=qkv_bias)  # 查询的线性变换
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)  # 键值的线性变换
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力dropout
        self.proj = nn.Linear(dim, dim)  # 输出投影
        self.proj_drop = nn.Dropout(proj_drop)  # 投影dropout

        self.sr_ratio = sr_ratio  # 空间减少比例
        if sr_ratio > 1:  # 如果使用空间减少
            self.sr = nn.Conv2d(  # 空间减少卷积
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)  # 卷积核大小和步长都等于减少比例
            self.norm = nn.LayerNorm(dim)  # 层归一化

    def execute(self, x, H, W):  # 前向传播函数，参数：输入张量x，高度H，宽度W
        B, N, C = x.shape  # 获取批次大小、序列长度、通道数
        q = self.q(x).reshape(B, N, self.num_heads, C //  # 生成查询并重塑为多头形式
                              self.num_heads).permute(0, 2, 1, 3)  # 调整维度顺序

        if self.sr_ratio > 1:  # 如果使用空间减少
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)  # 重塑为图像格式
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  # 空间减少后重塑
            x_ = self.norm(x_)  # 应用层归一化
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,  # 生成减少后的键值
                                     C // self.num_heads).permute(2, 0, 3, 1, 4)  # 调整维度
        else:  # 如果不使用空间减少
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C //  # 直接生成键值
                                    self.num_heads).permute(2, 0, 3, 1, 4)  # 调整维度
        k, v = kv[0], kv[1]  # 分离键和值

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # 计算注意力分数：Q * K^T * scale
        attn = nn.softmax(attn, dim=-1)  # 应用softmax
        attn = self.attn_drop(attn)  # 应用注意力dropout

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)  # 注意力加权：权重 * 值，然后重塑
        x = self.proj(x)  # 输出投影
        x = self.proj_drop(x)  # 应用投影dropout

        return x  # 返回输出


def main():  # 主函数，用于测试
    attention_block = EfficientAttention(64)  # 创建高效注意力模块，64维
    input = torch.rand([4, 128, 64])  # 创建随机输入：(批次=4, 序列=128, 维度=64)
    output = attention_block(input, 8, 8)  # 前向传播，假设8x8的空间尺寸
    print(input.size(), output.size())  # 打印输入和输出尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
