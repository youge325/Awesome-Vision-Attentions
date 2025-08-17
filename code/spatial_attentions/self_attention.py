import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块


class SelfAttention(nn.Module):  # 定义自注意力模块类
    """ self attention module"""  # 自注意力模块

    def __init__(self, in_dim):  # 初始化函数，参数：输入维度
        super(SelfAttention, self).__init__()  # 调用父类初始化
        self.chanel_in = in_dim  # 存储输入通道数

        self.query = nn.Conv(in_channels=in_dim,  # 查询卷积层（注意：应该是nn.Conv2d）
                             out_channels=in_dim, kernel_size=1)  # 1x1卷积生成查询
        self.key = nn.Conv(in_channels=in_dim,  # 键卷积层
                           out_channels=in_dim, kernel_size=1)  # 1x1卷积生成键
        self.value = nn.Conv(in_channels=in_dim,  # 值卷积层
                             out_channels=in_dim, kernel_size=1)  # 1x1卷积生成值

        self.softmax = nn.Softmax(dim=-1)  # Softmax函数

    def execute(self, x):  # 前向传播函数
        """
            inputs :  # 输入说明
                x : input feature maps( B X C X H X W)  # 输入特征图：(批次 x 通道 x 高度 x 宽度)
            returns :  # 返回说明
                out : attention value + input feature  # 输出：注意力值加输入特征
                attention: B X (HxW) X (HxW)  # 注意力权重：批次 x (高度x宽度) x (高度x宽度)
        """
        m_batchsize, C, height, width = x.size()  # 获取输入张量的形状
        proj_query = self.query(x).reshape(  # 生成查询并重塑形状
            m_batchsize, -1, width*height).transpose(0, 2, 1)  # (B, C, H*W) -> (B, H*W, C)
        proj_key = self.key(x).reshape(m_batchsize, -1, width*height)  # 生成键并重塑：(B, C, H*W)
        energy = nn.bmm(proj_query, proj_key)  # 计算能量：查询与键的矩阵乘法，结果(B, H*W, H*W)
        attention = self.softmax(energy)  # 对能量应用softmax得到注意力权重
        proj_value = self.value(x).reshape(m_batchsize, -1, width*height)  # 生成值并重塑：(B, C, H*W)

        out = nn.bmm(proj_value, attention.transpose(0, 2, 1))  # 注意力加权：值与转置后的注意力权重相乘
        out = out.reshape(m_batchsize, C, height, width)  # 重塑输出为原始空间形状

        return out  # 返回输出


def main():  # 主函数，用于测试
    attention_block = SelfAttention(64)  # 创建自注意力模块，64通道
    input = torch.rand([4, 64, 32, 32])  # 创建随机输入张量
    output = attention_block(input)  # 通过自注意力模块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
