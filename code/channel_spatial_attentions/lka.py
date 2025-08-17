# Visual Attention Network
# 视觉注意力网络
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块


class AttentionModule(nn.Module):  # 定义注意力模块类
    def __init__(self, dim):  # 初始化函数，参数：特征维度
        super().__init__()  # 调用父类初始化
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)  # 5x5深度卷积，填充2保持尺寸不变
        self.conv_spatial = nn.Conv2d(  # 空间注意力卷积
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)  # 7x7深度可分离卷积，膨胀系数3，填充9
        self.conv1 = nn.Conv2d(dim, dim, 1)  # 1x1卷积，用于特征融合

    def execute(self, x):  # 前向传播函数
        u = x.clone()  # 克隆输入作为残差项
        attn = self.conv0(x)  # 通过第一个卷积层
        attn = self.conv_spatial(attn)  # 通过空间注意力卷积（大感受野）
        attn = self.conv1(attn)  # 通过1x1卷积

        return u * attn  # 输入与注意力权重相乘


class SpatialAttention(nn.Module):  # 定义空间注意力类
    def __init__(self, d_model):  # 初始化函数，参数：模型维度
        super().__init__()  # 调用父类初始化

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)  # 第一个1x1投影卷积
        self.activation = nn.GELU()  # GELU激活函数
        self.spatial_gating_unit = AttentionModule(d_model)  # 空间门控单元（注意力模块）
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)  # 第二个1x1投影卷积

    def execute(self, x):  # 前向传播函数
        shorcut = x.clone()  # 保存输入用于残差连接
        x = self.proj_1(x)  # 通过第一个投影层
        x = self.activation(x)  # 应用GELU激活
        x = self.spatial_gating_unit(x)  # 通过空间门控单元
        x = self.proj_2(x)  # 通过第二个投影层
        x = x + shorcut  # 添加残差连接
        return x  # 返回结果


def main():  # 主函数，用于测试
    attention_block = SpatialAttention(64)  # 创建空间注意力模块，64个通道
    input = torch.rand([4, 64, 32, 32])  # 创建测试输入张量，形状为(4, 64, 32, 32)
    output = attention_block(input)  # 通过空间注意力模块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
