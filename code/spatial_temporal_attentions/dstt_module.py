# Decoupled spatial-temporal transformer for video inpainting (arXiv 2021)
# 用于视频修复的解耦时空变换器 (arXiv 2021)
import math  # 导入数学库
import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块


class Attention(nn.Module):  # 定义注意力类
    """
    Compute 'Scaled Dot Product Attention  # 计算缩放点积注意力
    """

    def __init__(self, p=0.1):  # 初始化函数，参数：dropout概率p
        super(Attention, self).__init__()  # 调用父类初始化
        self.dropout = nn.Dropout(p=p)  # Dropout层，用于防止过拟合

    def execute(self, query, key, value):  # 前向传播函数，参数：查询、键、值
        scores = torch.matmul(query, key.transpose(-2, -1)  # 计算注意力分数：Q * K^T
                           ) / math.sqrt(query.size(-1))  # 除以sqrt(d_k)进行缩放
        p_attn = nn.softmax(scores, dim=-1)  # 对分数应用softmax得到注意力权重
        p_attn = self.dropout(p_attn)  # 对注意力权重应用dropout
        p_val = torch.matmul(p_attn, value)  # 注意力权重与值相乘得到最终输出
        return p_val, p_attn  # 返回输出值和注意力权重


class MultiHeadedAttention(nn.Module):  # 定义多头注意力类
    """
    Take in model size and number of heads.  # 接受模型尺寸和注意力头数
    """

    def __init__(self, tokensize, d_model, head, mode, p=0.1):  # 初始化函数
        super().__init__()  # 调用父类初始化
        self.mode = mode  # 模式：'s'表示空间注意力，'t'表示时间注意力
        self.query_embedding = nn.Linear(d_model, d_model)  # 查询的线性变换层
        self.value_embedding = nn.Linear(d_model, d_model)  # 值的线性变换层
        self.key_embedding = nn.Linear(d_model, d_model)  # 键的线性变换层
        self.output_linear = nn.Linear(d_model, d_model)  # 输出的线性变换层
        self.attention = Attention(p=p)  # 注意力机制实例
        self.head = head  # 注意力头数
        self.h, self.w = tokensize  # token的高度和宽度

    def execute(self, x, t):  # 前向传播函数，参数：输入x，时间步长t
        bt, n, c = x.size()  # 获取输入尺寸：批次*时间，序列长度，通道数
        b = bt // t  # 计算批次大小
        c_h = c // self.head  # 计算每个头的通道数
        key = self.key_embedding(x)  # 生成键向量
        query = self.query_embedding(x)  # 生成查询向量
        value = self.value_embedding(x)  # 生成值向量
        if self.mode == 's':  # 如果是空间注意力模式
            key = key.view(b, t, n, self.head, c_h).permute(0, 1, 3, 2, 4)  # 重塑键：(b, t, head, n, c_h)
            query = query.view(b, t, n, self.head, c_h).permute(0, 1, 3, 2, 4)  # 重塑查询：(b, t, head, n, c_h)
            value = value.view(b, t, n, self.head, c_h).permute(0, 1, 3, 2, 4)  # 重塑值：(b, t, head, n, c_h)
            att, _ = self.attention(query, key, value)  # 计算空间注意力
            att = att.permute(0, 1, 3, 2, 4).view(bt, n, c)  # 重塑输出：(bt, n, c)
        elif self.mode == 't':  # 如果是时间注意力模式
            key = key.view(b, t, 2, self.h//2, 2, self.w//2, self.head, c_h)  # 重塑键为时空分块形式
            key = key.permute(0, 2, 4, 6, 1, 3, 5, 7).view(  # 调整维度顺序
                b, 4, self.head, -1, c_h)  # 重塑为(b, 4, head, -1, c_h)
            query = query.view(b, t, 2, self.h//2, 2,  # 重塑查询为时空分块形式
                               self.w//2, self.head, c_h)
            query = query.permute(0, 2, 4, 6, 1, 3, 5, 7).view(  # 调整维度顺序
                b, 4, self.head, -1, c_h)  # 重塑为(b, 4, head, -1, c_h)
            value = value.view(b, t, 2, self.h//2, 2,  # 重塑值为时空分块形式
                               self.w//2, self.head, c_h)
            value = value.permute(0, 2, 4, 6, 1, 3, 5, 7).view(  # 调整维度顺序
                b, 4, self.head, -1, c_h)  # 重塑为(b, 4, head, -1, c_h)
            att, _ = self.attention(query, key, value)  # 计算时间注意力
            att = att.view(b, 2, 2, self.head, t, self.h//2, self.w//2, c_h)  # 重塑注意力输出
            att = att.permute(0, 4, 1, 5, 2, 6, 3,  # 调整维度顺序
                              7).view(bt, n, c)  # 恢复原始形状(bt, n, c)
        output = self.output_linear(att)  # 通过输出线性层
        return output  # 返回最终输出


def main():  # 主函数，用于测试
    attention_block_s = MultiHeadedAttention(  # 创建空间注意力模块
        tokensize=[4, 8], d_model=64, head=4, mode='s')  # 参数：token尺寸4x8，模型维度64，4个头，空间模式
    attention_block_t = MultiHeadedAttention(  # 创建时间注意力模块
        tokensize=[4, 8], d_model=64, head=4, mode='t')  # 参数：token尺寸4x8，模型维度64，4个头，时间模式
    input = torch.rand([8, 32, 64])  # 创建随机输入：(批次*时间=8, 序列长度=32, 通道=64)
    output = attention_block_s(input, 2)  # 通过空间注意力，时间步长=2
    output = attention_block_t(output, 2)  # 通过时间注意力，时间步长=2
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
