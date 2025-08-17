# Strip Pooling: Rethinking spatial pooling for scene parsing (CVPR 2020)
# 条带池化：重新思考场景解析中的空间池化 (CVPR 2020)
import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块


class StripPooling(nn.Module):  # 定义条带池化类
    """
    Reference:  # 参考文献
    """

    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs):  # 初始化函数
        super(StripPooling, self).__init__()  # 调用父类初始化
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])  # 自适应平均池化1，使用第一个池化尺寸
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])  # 自适应平均池化2，使用第二个池化尺寸
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))  # 垂直条带池化，高度为1
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))  # 水平条带池化，宽度为1

        inter_channels = int(in_channels/4)  # 中间通道数为输入通道的1/4
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),  # 第一个1x1卷积序列
                                     norm_layer(inter_channels),  # 归一化层
                                     nn.ReLU())  # ReLU激活
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),  # 第二个1x1卷积序列
                                     norm_layer(inter_channels),  # 归一化层
                                     nn.ReLU())  # ReLU激活
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),  # 3x3卷积，padding=1
                                     norm_layer(inter_channels))  # 归一化层
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),  # 3x3卷积用于pool1路径
                                     norm_layer(inter_channels))  # 归一化层
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),  # 3x3卷积用于pool2路径
                                     norm_layer(inter_channels))  # 归一化层
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),  # 1x3卷积，水平条带
                                     norm_layer(inter_channels))  # 归一化层
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),  # 3x1卷积，垂直条带
                                     norm_layer(inter_channels))  # 归一化层
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),  # 第一分支的最后卷积
                                     norm_layer(inter_channels),  # 归一化层
                                     nn.ReLU())  # ReLU激活
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),  # 第二分支的最后卷积
                                     norm_layer(inter_channels),  # 归一化层
                                     nn.ReLU())  # ReLU激活
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),  # 融合两个分支的1x1卷积
                                   norm_layer(in_channels))  # 归一化层
        # bilinear interpolate options
        self._up_kwargs = up_kwargs  # 上采样参数配置

    def execute(self, x):  # 前向传播函数
        _, _, h, w = x.size()  # 获取输入的高度和宽度
        x1 = self.conv1_1(x)  # 第一个分支的1x1卷积
        x2 = self.conv1_2(x)  # 第二个分支的1x1卷积
        x2_1 = self.conv2_0(x1)  # 直接通过3x3卷积的特征
        x2_2 = nn.interpolate(self.conv2_1(self.pool1(x1)),  # pool1路径：池化->卷积->上采样
                              (h, w), **self._up_kwargs)  # 双线性插值上采样到原始尺寸
        x2_3 = nn.interpolate(self.conv2_2(self.pool2(x1)),  # pool2路径：池化->卷积->上采样
                              (h, w), **self._up_kwargs)  # 双线性插值上采样到原始尺寸
        x2_4 = nn.interpolate(self.conv2_3(self.pool3(x2)),  # 水平条带路径：垂直池化->1x3卷积->上采样
                              (h, w), **self._up_kwargs)  # 双线性插值上采样到原始尺寸
        x2_5 = nn.interpolate(self.conv2_4(self.pool4(x2)),  # 垂直条带路径：水平池化->3x1卷积->上采样
                              (h, w), **self._up_kwargs)  # 双线性插值上采样到原始尺寸
        x1 = self.conv2_5(nn.relu(x2_1 + x2_2 + x2_3))  # 第一分支：三个特征相加，ReLU激活，再卷积
        x2 = self.conv2_6(nn.relu(x2_5 + x2_4))  # 第二分支：两个条带特征相加，ReLU激活，再卷积
        out = self.conv3(torch.concat([x1, x2], dim=1))  # 沿通道维度连接两个分支，通过1x1卷积融合
        return nn.relu(x + out)  # 残差连接：输入加上输出，ReLU激活


def main():  # 主函数，用于测试
    attention_block = StripPooling(  # 创建条带池化模块
        64, (20, 12), nn.BatchNorm2d, {'mode': 'bilinear', 'align_corners': True})  # 参数：64通道，池化尺寸，批量归一化，双线性插值
    input = torch.rand([4, 64, 32, 32])  # 创建随机输入张量
    output = attention_block(input)  # 通过条带池化模块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
