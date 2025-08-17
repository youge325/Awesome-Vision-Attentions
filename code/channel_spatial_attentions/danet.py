import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块


class DANetHead(nn.Module):  # 定义DANet头部模块类

    def __init__(self, in_channels, out_channels):  # 初始化函数，参数：输入通道数、输出通道数
        super(DANetHead, self).__init__()  # 调用父类初始化
        inter_channels = in_channels // 4  # 中间层通道数为输入通道数的1/4
        self.conv5a = nn.Sequential(nn.Conv(in_channels, inter_channels, 3, padding=1, bias=False),  # 位置注意力分支的卷积序列
                                    nn.BatchNorm(inter_channels),  # 批归一化
                                    nn.ReLU())  # ReLU激活函数

        self.conv5c = nn.Sequential(nn.Conv(in_channels, inter_channels, 3, padding=1, bias=False),  # 通道注意力分支的卷积序列
                                    nn.BatchNorm(inter_channels),  # 批归一化
                                    nn.ReLU())  # ReLU激活函数

        self.sa = PAM_Module(inter_channels)  # 位置注意力模块
        self.sc = CAM_Module(inter_channels)  # 通道注意力模块
        self.conv51 = nn.Sequential(nn.Conv(inter_channels, inter_channels, 3, padding=1, bias=False),  # 位置注意力后的卷积序列
                                    nn.BatchNorm(inter_channels),  # 批归一化
                                    nn.ReLU())  # ReLU激活函数
        self.conv52 = nn.Sequential(nn.Conv(inter_channels, inter_channels, 3, padding=1, bias=False),  # 通道注意力后的卷积序列
                                    nn.BatchNorm(inter_channels),  # 批归一化
                                    nn.ReLU())  # ReLU激活函数

        self.conv8 = nn.Sequential(nn.Dropout(  # 最终输出的卷积序列
            0.1, False), nn.Conv(inter_channels, out_channels, 1))  # Dropout后接1x1卷积

    def execute(self, x):  # 前向传播函数

        feat1 = self.conv5a(x)  # 通过位置注意力分支的预处理卷积
        sa_feat = self.sa(feat1)  # 通过位置注意力模块
        sa_conv = self.conv51(sa_feat)  # 位置注意力后的后处理卷积

        feat2 = self.conv5c(x)  # 通过通道注意力分支的预处理卷积
        sc_feat = self.sc(feat2)  # 通过通道注意力模块
        sc_conv = self.conv52(sc_feat)  # 通道注意力后的后处理卷积

        feat_sum = sa_conv+sc_conv  # 将位置注意力和通道注意力的结果相加

        sasc_output = self.conv8(feat_sum)  # 通过最终的输出卷积

        return sasc_output  # 返回DANet的输出


class PAM_Module(nn.Module):  # 定义位置注意力模块类
    """ Position attention module"""  # 位置注意力模块
    # Ref from SAGAN  # 参考SAGAN

    def __init__(self, in_dim):  # 初始化函数，参数：输入维度
        super(PAM_Module, self).__init__()  # 调用父类初始化
        self.chanel_in = in_dim  # 存储输入通道数

        self.query_conv = nn.Conv(  # 查询卷积层
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)  # 1x1卷积，输出通道为输入的1/8
        self.key_conv = nn.Conv(  # 键卷积层
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)  # 1x1卷积，输出通道为输入的1/8
        self.value_conv = nn.Conv(  # 值卷积层
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # 1x1卷积，输出通道与输入相同
        self.gamma = torch.zeros(1)  # 初始化缩放参数为0

        self.softmax = nn.Softmax(dim=-1)  # Softmax函数，在最后一个维度上

    def execute(self, x):  # 前向传播函数
        """
            inputs :  输入：
                x : input feature maps( B X C X H X W)  输入特征图(B X C X H X W)
            returns :  返回：
                out : attention value + input feature  注意力值 + 输入特征
                attention: B X (HxW) X (HxW)  注意力权重: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()  # 获取输入张量的形状
        proj_query = self.query_conv(x).reshape(  # 生成查询并重塑
            m_batchsize, -1, width*height).transpose(0, 2, 1)  # 形状：(B, HW, C//8)
        proj_key = self.key_conv(x).reshape(m_batchsize, -1, width*height)  # 生成键并重塑，形状：(B, C//8, HW)
        energy = nn.bmm(proj_query, proj_key)  # 计算注意力能量：(B, HW, HW)
        attention = self.softmax(energy)  # 对注意力能量应用softmax
        proj_value = self.value_conv(x).reshape(m_batchsize, -1, width*height)  # 生成值并重塑，形状：(B, C, HW)

        out = nn.bmm(proj_value, attention.transpose(0, 2, 1))  # 应用注意力权重：(B, C, HW)
        out = out.reshape(m_batchsize, C, height, width)  # 重塑回原始特征图形状

        out = self.gamma*out + x  # 缩放注意力输出并添加残差连接
        return out  # 返回结果


class CAM_Module(nn.Module):  # 定义通道注意力模块类
    """ Channel attention module"""  # 通道注意力模块

    def __init__(self, in_dim):  # 初始化函数，参数：输入维度
        super(CAM_Module, self).__init__()  # 调用父类初始化
        self.chanel_in = in_dim  # 存储输入通道数
        self.gamma = torch.zeros(1)  # 初始化缩放参数为0
        self.softmax = nn.Softmax(dim=-1)  # Softmax函数，在最后一个维度上

    def execute(self, x):  # 前向传播函数
        """
            inputs :  输入：
                x : input feature maps( B X C X H X W)  输入特征图(B X C X H X W)
            returns :  返回：
                out : attention value + input feature  注意力值 + 输入特征
                attention: B X C X C  注意力权重: B X C X C
        """
        m_batchsize, C, height, width = x.size()  # 获取输入张量的形状
        proj_query = x.reshape(m_batchsize, C, -1)  # 重塑查询，形状：(B, C, HW)
        proj_key = x.reshape(m_batchsize, C, -1).transpose(0, 2, 1)  # 重塑键并转置，形状：(B, HW, C)
        energy = nn.bmm(proj_query, proj_key)  # 计算通道间的注意力能量：(B, C, C)
        #energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy  # 注释掉的能量归一化
        attention = self.softmax(energy)  # 对注意力能量应用softmax
        proj_value = x.reshape(m_batchsize, C, -1)  # 重塑值，形状：(B, C, HW)

        out = nn.bmm(attention, proj_value)  # 应用通道注意力权重：(B, C, HW)
        out = out.reshape(m_batchsize, C, height, width)  # 重塑回原始特征图形状

        out = self.gamma*out + x  # 缩放注意力输出并添加残差连接
        return out  # 返回结果


def main():  # 主函数，用于测试
    attention_block = DANetHead(64, 64)  # 创建DANet头部模块，输入输出都是64通道
    input = torch.rand([4, 64, 32, 32])  # 创建测试输入张量，形状为(4, 64, 32, 32)
    output = attention_block(input)  # 通过DANet模块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
