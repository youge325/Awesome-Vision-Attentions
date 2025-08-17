# Dynamic convolution: Attention over convolution kernels (CVPR 2020)
# 动态卷积：对卷积核的注意力机制 (CVPR 2020)
import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块


class attention2d(nn.Module):  # 定义2D注意力模块类
    def __init__(self, in_planes, ratios, K, temperature):  # 初始化函数，参数：输入通道数、比例、卷积核数量、温度参数
        super(attention2d, self).__init__()  # 调用父类初始化
        # for reducing τ temperature from 30 to 1 linearly in the first 10 epochs.
        # 用于在前10个epoch中将温度参数τ从30线性降低到1
        assert temperature % 3 == 1  # 断言温度参数必须满足条件（除以3余1）
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，输出尺寸为1x1

        if in_planes != 3:  # 如果输入通道不是3（非RGB图像）
            hidden_planes = int(in_planes * ratios) + 1  # 计算隐藏层通道数
        else:  # 如果输入通道是3（RGB图像）
            hidden_planes = K  # 隐藏层通道数等于卷积核数量

        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)  # 第一个1x1卷积层，无偏置
        # self.relu  = nn.ReLU()  # 注释掉的ReLU激活函数
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)  # 第二个1x1卷积层，有偏置，输出K个通道
        self.temperature = temperature  # 存储温度参数

    def update__temperature(self):  # 更新温度参数的函数
        if self.temperature != 1:  # 如果温度参数不等于1
            self.temperature -= 3  # 温度参数减3

    def execute(self, z):  # 前向传播函数
        z = self.avgpool(z)  # 全局平均池化，将特征图压缩为1x1
        z = self.fc1(z)  # 通过第一个1x1卷积层
        # z = self.relu(z)  # 注释掉的ReLU激活
        z = nn.relu(z)  # 使用函数式ReLU激活
        z = self.fc2(z)  # 通过第二个1x1卷积层，得到K个注意力权重
        z = z.view(z.size(0), -1)  # 将张量reshape为(batch_size, K)
        # z = self.fc2(z).view(z.size(0), -1)  # 注释掉的合并操作

        return nn.softmax(z/self.temperature, 1)  # 使用温度参数缩放后进行softmax，得到注意力权重


class Dynamic_conv2d(nn.Module):  # 定义动态2D卷积类
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4, temperature=34):  # 初始化函数，包含所有卷积参数
        super(Dynamic_conv2d, self).__init__()  # 调用父类初始化

        if in_planes % groups != 0:  # 检查输入通道数是否能被组数整除
            raise ValueError('Error : in_planes%groups != 0')  # 抛出错误
        self.in_planes = in_planes  # 存储输入通道数
        self.out_planes = out_planes  # 存储输出通道数
        self.kernel_size = kernel_size  # 存储卷积核大小
        self.stride = stride  # 存储步长
        self.padding = padding  # 存储填充
        self.dilation = dilation  # 存储膨胀系数
        self.groups = groups  # 存储分组数
        self.bias = bias  # 存储是否使用偏置
        self.K = K  # 存储卷积核数量
        self.attention = attention2d(in_planes, ratio, K, temperature)  # 创建注意力模块
        self.weight = torch.randn((  # 初始化K个卷积核权重，形状为(K, out_planes, in_planes//groups, kernel_size, kernel_size)
            K, out_planes, in_planes//groups, kernel_size, kernel_size))

        if bias:  # 如果使用偏置
            self.bias = torch.randn((K, out_planes))  # 初始化K个偏置参数
        else:  # 如果不使用偏置
            self.bias = None  # 设置偏置为None

    def update_temperature(self):  # 更新温度参数的函数
        self.attention.update__temperature()  # 调用注意力模块的温度更新函数

    def execute(self, z):  # 前向传播函数

        #         Regard batch as a dimensional variable, perform group convolution,
        #         because the weight of group convolution is different,
        #         and the weight of dynamic convolution is also different
        #         将批次作为维度变量，执行分组卷积，
        #         因为分组卷积的权重不同，动态卷积的权重也不同
        softmax_attention = self.attention(z)  # 通过注意力模块获取K个卷积核的注意力权重
        batch_size, in_planes, height, width = z.size()  # 获取输入张量的维度
        # changing into dimension for group convolution
        # 改变维度以进行分组卷积
        z = z.view(1, -1, height, width)  # 将批次维度合并到通道维度，形状变为(1, batch_size*in_planes, H, W)
        weight = self.weight.view(self.K, -1)  # 将卷积核权重reshape为(K, out_planes*in_planes*kernel_size*kernel_size)

#         The generation of the weight of dynamic convolution,
#         which generates batch_size convolution parameters
#         (each parameter is different)
#         生成动态卷积的权重，
#         生成batch_size个卷积参数（每个参数都不同）
        aggregate_weight = torch.matmul(softmax_attention, weight).view(-1, self.in_planes,  # 使用注意力权重对K个卷积核进行加权平均
                                                                     self.kernel_size, self.kernel_size)  # expects two matrices (2D tensors)  # 期望两个2D张量进行矩阵乘法
        if self.bias is not None:  # 如果使用偏置
            aggregate_bias = torch.matmul(softmax_attention, self.bias).view(-1)  # 使用注意力权重对K个偏置进行加权平均
            output = nn.conv2d(z, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,  # 执行卷积操作，使用聚合后的权重和偏置
                               dilation=self.dilation, groups=self.groups * batch_size)  # 分组数为原分组数乘以批次大小
        else:  # 如果不使用偏置
            output = nn.conv2d(z, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,  # 执行卷积操作，不使用偏置
                               dilation=self.dilation, groups=self.groups * batch_size)  # 分组数为原分组数乘以批次大小
        output = output.view(batch_size, self.out_planes,  # 将输出张量恢复为正确的批次维度
                             output.size(-2), output.size(-1))  # 形状为(batch_size, out_planes, H_out, W_out)
        # print('2d-att-for')  # 注释掉的调试信息
        return output  # 返回输出张量


def main():  # 主函数，用于测试
    attention_block = Dynamic_conv2d(64, 64, 3, padding=1)  # 创建一个动态卷积块，输入输出64通道，3x3卷积核，填充1
    input = torch.ones([4, 64, 32, 32])  # 创建测试输入张量，形状为(4, 64, 32, 32)
    output = attention_block(input)  # 通过动态卷积块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
