# Attention augmented convolutional networks (ICCV 2019)
# 注意力增强卷积网络 (ICCV 2019)
import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块


class AugmentedConv(nn.Module):  # 定义注意力增强卷积类
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, relative):  # 初始化函数
        super(AugmentedConv, self).__init__()  # 调用父类初始化
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.kernel_size = kernel_size  # 卷积核大小
        self.dk = dk  # 键和查询的维度
        self.dv = dv  # 值的维度
        self.Nh = Nh  # 注意力头的数量
        self.relative = relative  # 是否使用相对位置编码

        self.conv_out = nn.Conv2d(  # 传统卷积输出分支
            self.in_channels, self.out_channels - self.dv, self.kernel_size, padding=1)  # 输出通道为总输出减去注意力输出

        self.qkv_conv = nn.Conv2d(  # 生成查询、键、值的1x1卷积
            self.in_channels, 2 * self.dk + self.dv, kernel_size=1)  # 输出通道为2*dk+dv

        self.attn_out = nn.Conv2d(self.dv, self.dv, 1)  # 注意力输出的1x1卷积

    def execute(self, x):  # 前向传播函数
        # Input x
        # (batch_size, channels, height, width)
        batch, _, height, width = x.size()  # 获取输入张量的批次大小、高度和宽度

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)  # 传统卷积分支的输出

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(  # 计算扁平化的查询、键、值张量
            x, self.dk, self.dv, self.Nh)  # 传入输入张量和相关参数
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)  # 计算注意力分数：Q^T * K

        if self.relative:  # 如果使用相对位置编码
            h_rel_logits, w_rel_logits = self.relative_logits(q)  # 计算高度和宽度的相对位置logits
            logits += h_rel_logits  # 加上高度相对位置logits
            logits += w_rel_logits  # 加上宽度相对位置logits
        weights = nn.softmax(logits, dim=-1)  # 对logits应用softmax得到注意力权重

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))  # 注意力输出：权重矩阵乘以值向量
        attn_out = torch.reshape(  # 重塑注意力输出的形状
            attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))  # 恢复到(batch, Nh, dv/Nh, H, W)形状
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)  # 合并多个注意力头
        attn_out = self.attn_out(attn_out)  # 通过输出卷积层
        return torch.concat((conv_out, attn_out), dim=1)  # 沿通道维度连接卷积输出和注意力输出

    def compute_flat_qkv(self, x, dk, dv, Nh):  # 计算扁平化的查询、键、值函数
        N, _, H, W = x.size()  # 获取输入张量的形状
        qkv = self.qkv_conv(x)  # 通过1x1卷积生成QKV
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)  # 沿通道维度分割成查询、键、值
        q = self.split_heads_2d(q, Nh)  # 将查询分割成多头
        k = self.split_heads_2d(k, Nh)  # 将键分割成多头
        v = self.split_heads_2d(v, Nh)  # 将值分割成多头

        dkh = dk // Nh  # 每个头的键/查询维度
        q *= dkh ** -0.5  # 对查询进行缩放，防止梯度消失
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))  # 扁平化查询张量
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))  # 扁平化键张量
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))  # 扁平化值张量
        return flat_q, flat_k, flat_v, q, k, v  # 返回扁平化和原始的QKV张量

    def split_heads_2d(self, x, Nh):  # 将张量分割成多个注意力头
        batch, channels, height, width = x.size()  # 获取输入张量的形状
        ret_shape = (batch, Nh, channels // Nh, height, width)  # 计算输出形状
        split = torch.reshape(x, ret_shape)  # 重塑张量形状
        return split  # 返回分割后的张量

    def combine_heads_2d(self, x):  # 合并多个注意力头
        batch, Nh, dv, H, W = x.size()  # 获取输入张量的形状
        ret_shape = (batch, Nh * dv, H, W)  # 计算输出形状：合并头数和值维度
        return torch.reshape(x, ret_shape)  # 重塑并返回张量

    def relative_logits(self, q):  # 计算相对位置编码的logits
        B, Nh, dk, H, W = q.size()  # 获取查询张量的形状
        q = torch.transpose(q, 2, 4).transpose(2, 3)  # 调整查询张量的维度顺序

        key_rel_w = torch.randn((2 * W - 1, dk))  # 宽度方向的相对位置编码键
        rel_logits_w = self.relative_logits_1d(q, key_rel_w, H, W, Nh, "w")  # 计算宽度方向相对位置logits

        key_rel_h = torch.randn((2 * H - 1, dk))  # 高度方向的相对位置编码键
        rel_logits_h = self.relative_logits_1d(  # 计算高度方向相对位置logits
            torch.transpose(q, 2, 3), key_rel_h, W, H, Nh, "h")  # 转置查询张量的高度和宽度维度

        return rel_logits_h, rel_logits_w  # 返回高度和宽度的相对位置logits

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):  # 计算一维相对位置logits
        rel_logits = torch.matmul(q, rel_k.transpose(0, 1))  # 查询与相对位置键的矩阵乘法
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))  # 重塑logits形状
        rel_logits = self.rel_to_abs(rel_logits)  # 将相对位置转换为绝对位置

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))  # 重塑为5维张量
        rel_logits = torch.unsqueeze(rel_logits, dim=3)  # 在第3维增加一个维度
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))  # 在高度维度重复

        if case == "w":  # 如果是宽度方向
            rel_logits = torch.transpose(rel_logits, 3, 4)  # 转置第3和第4维
        elif case == "h":  # 如果是高度方向
            rel_logits = torch.transpose(  # 进行多次转置操作
                rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)  # 调整维度顺序
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))  # 重塑为最终形状
        return rel_logits  # 返回相对位置logits

    def rel_to_abs(self, x):  # 将相对位置转换为绝对位置
        B, Nh, L, _ = x.size()  # 获取输入张量的形状

        col_pad = torch.zeros((B, Nh, L, 1))  # 创建列填充张量
        x = torch.concat((x, col_pad), dim=3)  # 在最后一维连接填充

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))  # 扁平化张量
        flat_pad = torch.zeros((B, Nh, L - 1))  # 创建扁平填充张量
        flat_x_padded = torch.concat((flat_x, flat_pad), dim=2)  # 连接扁平填充

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))  # 重塑为最终形状
        final_x = final_x[:, :, :L, L - 1:]  # 截取所需部分
        return final_x  # 返回绝对位置张量


def main():  # 主函数，用于测试
    attention_block = AugmentedConv(64, 64, 3, 40, 4, 4, True)  # 创建注意力增强卷积模块
    input = torch.rand([4, 64, 32, 32])  # 创建随机输入张量
    output = attention_block(input)  # 通过模块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
