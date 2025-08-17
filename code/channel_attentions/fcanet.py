# Fcanet: Frequency channel attention networks (ICCV 2021)
# FCANet：频域通道注意力网络 (ICCV 2021)
import math  # 导入数学库
import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块


def get_freq_indices(method):  # 获取频率索引的函数
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',  # 断言方法必须在指定列表中
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',  # 支持顶部、底部、低频不同数量的频率选择
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])  # 提取频率数量（方法名后面的数字）
    if 'top' in method:  # 如果是顶部频率方法
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0,  # 预定义的顶部频率x坐标索引
                             0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6,  # 预定义的顶部频率y坐标索引
                             3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
        mapper_x = all_top_indices_x[:num_freq]  # 根据需要的频率数量选择x坐标
        mapper_y = all_top_indices_y[:num_freq]  # 根据需要的频率数量选择y坐标
    elif 'low' in method:  # 如果是低频方法
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0,  # 预定义的低频x坐标索引
                             1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4,  # 预定义的低频y坐标索引
                             3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
        mapper_x = all_low_indices_x[:num_freq]  # 根据需要的频率数量选择x坐标
        mapper_y = all_low_indices_y[:num_freq]  # 根据需要的频率数量选择y坐标
    elif 'bot' in method:  # 如果是底部频率方法
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4,  # 预定义的底部频率x坐标索引
                             6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2,  # 预定义的底部频率y坐标索引
                             2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]
        mapper_x = all_bot_indices_x[:num_freq]  # 根据需要的频率数量选择x坐标
        mapper_y = all_bot_indices_y[:num_freq]  # 根据需要的频率数量选择y坐标
    else:  # 如果不是支持的方法
        raise NotImplementedError  # 抛出未实现错误
    return mapper_x, mapper_y  # 返回x和y坐标映射


class MultiSpectralAttentionLayer(nn.Module):  # 定义多光谱注意力层类
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):  # 初始化函数
        super(MultiSpectralAttentionLayer, self).__init__()  # 调用父类初始化
        self.reduction = reduction  # 存储降维比例
        self.dct_h = dct_h  # 存储DCT高度
        self.dct_w = dct_w  # 存储DCT宽度

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)  # 获取频率索引映射
        self.num_split = len(mapper_x)  # 存储分割数量（频率数量）
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]  # 将x坐标映射到实际DCT尺寸
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]  # 将y坐标映射到实际DCT尺寸
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # 使不同尺寸的频率与7x7频率空间相同
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7
        # 例如，14x14中的(2,2)等同于7x7中的(1,1)

        self.dct_layer = MultiSpectralDCTLayer(  # 创建多光谱DCT层
            dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(  # 创建全连接序列
            nn.Linear(channel, channel // reduction, bias=False),  # 降维线性层
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(channel // reduction, channel, bias=False),  # 升维线性层
            nn.Sigmoid()  # Sigmoid激活函数，生成注意力权重
        )
        self.avgpool = nn.AdaptiveAvgPool2d((self.dct_h, self.dct_w))  # 自适应平均池化到DCT尺寸

    def execute(self, x):  # 前向传播函数
        n, c, h, w = x.shape  # 获取输入张量的形状
        x_pooled = x  # 初始化池化后的张量
        if h != self.dct_h or w != self.dct_w:  # 如果输入尺寸与DCT尺寸不匹配
            x_pooled = self.avgpool(x)  # 进行自适应平均池化
            # If you have concerns about one-line-change, don't worry.   :)
            # 如果你担心这一行的改动，不用担心 :)
            # In the ImageNet models, this line will never be triggered.
            # 在ImageNet模型中，这行代码永远不会被触发
            # This is for compatibility in instance segmentation and object detection.
            # 这是为了与实例分割和目标检测兼容
        y = self.dct_layer(x_pooled)  # 通过DCT层处理池化后的特征

        y = self.fc(y).view(n, c, 1, 1)  # 通过全连接层生成注意力权重并重塑形状
        return x * y.expand_as(x)  # 将注意力权重扩展并与输入特征相乘


class MultiSpectralDCTLayer(nn.Module):  # 定义多光谱DCT层类
    """
    Generate dct filters  生成DCT滤波器
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):  # 初始化函数
        super(MultiSpectralDCTLayer, self).__init__()  # 调用父类初始化

        assert len(mapper_x) == len(mapper_y)  # 断言x和y映射长度相同
        assert channel % len(mapper_x) == 0  # 断言通道数能被映射长度整除

        self.num_freq = len(mapper_x)  # 存储频率数量

        # fixed DCT init  # 固定DCT初始化
        self.weight = self.get_dct_filter(  # 获取DCT滤波器权重
            height, width, mapper_x, mapper_y, channel)

    def execute(self, x):  # 前向传播函数
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))  # 断言输入是4维张量
        # n, c, h, w = x.shape  # 注释掉的形状获取

        x = x * self.weight  # 将输入与DCT权重相乘
        result = torch.sum(torch.sum(x, dim=2), dim=2)  # 在高度和宽度维度上求和
        return result  # 返回结果

    def build_filter(self, pos, freq, POS):  # 构建DCT滤波器的函数
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)  # 计算DCT基函数值
        if freq == 0:  # 如果频率为0（DC分量）
            return result  # 直接返回结果
        else:  # 如果频率不为0
            return result * math.sqrt(2)  # 乘以sqrt(2)进行归一化

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):  # 获取DCT滤波器的函数
        dct_filter = torch.zeros((channel, tile_size_x, tile_size_y))  # 初始化DCT滤波器张量

        c_part = channel // len(mapper_x)  # 计算每个频率分量对应的通道数

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):  # 遍历每个频率分量
            for t_x in range(tile_size_x):  # 遍历x方向的每个位置
                for t_y in range(tile_size_y):  # 遍历y方向的每个位置
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(  # 为对应通道设置DCT滤波器值
                        t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)  # x和y方向DCT基函数的乘积

        return dct_filter  # 返回DCT滤波器


def main():  # 主函数，用于测试
    attention_block = MultiSpectralAttentionLayer(64, 16, 16)  # 创建多光谱注意力层，64通道，16x16 DCT尺寸
    input = torch.ones([4, 64, 32, 32])  # 创建测试输入张量，形状为(4, 64, 32, 32)
    output = attention_block(input)  # 通过多光谱注意力层进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
