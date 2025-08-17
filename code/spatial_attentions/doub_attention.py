# A2-Nets: Double Attention Networks (NIPS 2018)
# A2网络：双重注意力网络 (NIPS 2018)
import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块


class DoubleAtten(nn.Module):  # 定义双重注意力类
    def __init__(self, in_c):  # 初始化函数，参数：输入通道数
        super(DoubleAtten, self).__init__()  # 调用父类初始化
        self.in_c = in_c  # 存储输入通道数
        self.convA = nn.Conv2d(in_c, in_c, kernel_size=1)  # 用于生成特征图A的1x1卷积
        self.convB = nn.Conv2d(in_c, in_c, kernel_size=1)  # 用于生成注意力图B的1x1卷积
        self.convV = nn.Conv2d(in_c, in_c, kernel_size=1)  # 用于生成值V的1x1卷积

    def execute(self, input):  # 前向传播函数

        feature_maps = self.convA(input)  # 通过卷积A生成特征图
        atten_map = self.convB(input)  # 通过卷积B生成注意力图
        b, _, h, w = feature_maps.shape  # 获取特征图的形状：批次、通道、高度、宽度

        feature_maps = feature_maps.view(b, 1, self.in_c, h*w)  # 重塑特征图：(b, 1, c, h*w)
        atten_map = atten_map.view(b, self.in_c, 1, h*w)  # 重塑注意力图：(b, c, 1, h*w)
        global_descriptors = torch.mean(  # 计算全局描述符
            (feature_maps * nn.softmax(atten_map, dim=-1)), dim=-1)  # 特征图乘以softmax注意力权重后求平均

        v = self.convV(input)  # 通过卷积V生成值向量
        atten_vectors = nn.softmax(  # 计算注意力向量
            v.view(b, self.in_c, h*w), dim=-1)  # 对值向量应用softmax
        out = nn.bmm(atten_vectors.permute(0, 2, 1),  # 批次矩阵乘法：注意力向量转置
                     global_descriptors).permute(0, 2, 1)  # 乘以全局描述符后再转置

        return out.view(b, _, h, w)  # 将输出重塑回原始形状并返回


def main():  # 主函数，用于测试
    attention_block = DoubleAtten(64)  # 创建双重注意力模块，64通道
    input = torch.rand([4, 64, 32, 32])  # 创建随机输入张量
    output = attention_block(input)  # 通过双重注意力模块进行前向传播
    torch.autograd.grad(output, input)  # 计算梯度（需要设置requires_grad=True才能正常运行）
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
