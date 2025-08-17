import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
from torch import init  # 导入初始化模块

class OCRHead(nn.Module):  # 定义OCR头部模块类
    def __init__(self, in_channels, n_cls=19):  # 初始化函数，参数：输入通道数，类别数（默认19）
        super(OCRHead, self).__init__()  # 调用父类初始化
        self.relu = nn.ReLU()  # ReLU激活函数
        self.in_channels = in_channels  # 存储输入通道数
        self.softmax = nn.Softmax(dim = 2)  # Softmax函数，沿第2维应用
        self.conv_1x1 = nn.Conv(in_channels, in_channels, kernel_size=1)  # 1x1卷积层（注意：应该是nn.Conv2d）
        self.last_conv = nn.Conv(in_channels * 2, n_cls, kernel_size=3, stride=1, padding=1)  # 最后的3x3卷积层
        self._zero_init_conv()  # 零初始化卷积权重
    def _zero_init_conv(self):  # 零初始化卷积权重函数
        self.conv_1x1.weight = init.constant([self.in_channels, self.in_channels, 1, 1], 'float', value=0.0)  # 将1x1卷积权重初始化为0

    def execute(self, context, feature):  # 前向传播函数，参数：上下文和特征
        batch_size, c, h, w = feature.shape  # 获取特征图的形状
        origin_feature = feature  # 保存原始特征图
        feature = feature.reshape(batch_size, c, -1).transpose(0, 2, 1)  # 重塑特征：(b, c, h*w) -> (b, h*w, c)
        context = context.reshape(batch_size, context.shape[1], -1)  # 重塑上下文：(b, n_cls, h*w)
        attention = self.softmax(context)  # 对上下文应用softmax得到注意力权重
        ocr_context = nn.bmm(attention, feature).transpose(0, 2, 1)  # 计算OCR上下文：注意力加权特征，结果(b, c, n_cls)
        relation = nn.bmm(feature, ocr_context).transpose(0, 2, 1)  # 计算关系：特征与OCR上下文的关系，结果(b, n_cls, h*w)
        attention = self.softmax(relation)  # 对关系应用softmax得到新的注意力权重
        result = nn.bmm(ocr_context, attention).reshape(batch_size, c, h, w)  # 最终结果：OCR上下文与注意力的加权组合，重塑为原始尺寸
        result = self.conv_1x1(result)  # 通过1x1卷积处理结果
        result = torch.concat ([result, origin_feature], dim=1)  # 沿通道维度连接处理后的结果和原始特征
        result = self.last_conv (result)  # 通过最后的卷积层得到最终输出
        return result  # 返回最终结果




