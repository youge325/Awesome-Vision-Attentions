# Context encoding for semantic segmentation (CVPR 2018)
# 语义分割的上下文编码 (CVPR 2018)
import torch  # 导入PyTorch库
from torch import nn, init  # 导入神经网络模块和初始化模块


class Encoding(nn.Module):  # 定义编码模块类
    def __init__(self, channels, num_codes):  # 初始化函数，参数：通道数、码本数量
        super(Encoding, self).__init__()  # 调用父类初始化
        # init codewords and smoothing factor
        # 初始化码本和平滑因子
        self.channels, self.num_codes = channels, num_codes  # 存储通道数和码本数量
        std = 1. / ((num_codes * channels)**0.5)  # 计算标准差，用于初始化
        # [num_codes, channels]
        # [码本数量, 通道数]
        self.codewords = init.uniform_(  # 初始化码本，使用均匀分布
            torch.randn((num_codes, channels)), -std, std)  # 形状为(码本数量, 通道数)
        # [num_codes]
        # [码本数量]
        self.scale = init.uniform_(torch.randn((num_codes,)), -1, 0)  # 初始化尺度参数，范围[-1, 0]

    @staticmethod  # 静态方法装饰器
    def scaled_l2(x, codewords, scale):  # 计算缩放L2距离的静态方法
        num_codes, channels = codewords.size()  # 获取码本的形状
        batch_size = x.size(0)  # 获取批次大小
        reshaped_scale = scale.view((1, 1, num_codes))  # 重塑尺度参数为广播形状
        expanded_x = x.unsqueeze(2).expand(  # 在第2维增加维度并扩展
            (batch_size, x.size(1), num_codes, channels))  # 扩展为(批次, 空间位置, 码本数量, 通道)
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))  # 重塑码本为广播形状

        scaled_l2_norm = reshaped_scale * (  # 计算缩放L2距离
            expanded_x - reshaped_codewords).pow(2).sum(dim=3)  # (x - 码本)^2在通道维度求和，再乘以尺度
        return scaled_l2_norm  # 返回缩放L2距离

    @staticmethod  # 静态方法装饰器
    def aggregate(assignment_weights, x, codewords):  # 聚合特征的静态方法
        num_codes, channels = codewords.size()  # 获取码本的形状
        reshaped_codewords = codewords.view((1, 1, num_codes, channels))  # 重塑码本为广播形状
        batch_size = x.size(0)  # 获取批次大小

        expanded_x = x.unsqueeze(2).expand(  # 在第2维增加维度并扩展
            (batch_size, x.size(1), num_codes, channels))  # 扩展为(批次, 空间位置, 码本数量, 通道)
        encoded_feat = (assignment_weights.unsqueeze(3) *  # 分配权重增加一个维度
                        (expanded_x - reshaped_codewords)).sum(dim=1)  # 加权求和(x - 码本)在空间维度
        return encoded_feat  # 返回编码特征

    def execute(self, x):  # 前向传播函数
        assert x.ndim == 4 and x.size(1) == self.channels  # 断言输入是4D张量且通道数匹配
        # [batch_size, channels, height, width]
        # [批次大小, 通道数, 高度, 宽度]
        batch_size = x.size(0)  # 获取批次大小
        # [batch_size, height x width, channels]
        # [批次大小, 高度x宽度, 通道数]
        x = x.view(batch_size, self.channels, -1).transpose(0, 2, 1)  # 重塑并转置：(批次, 通道, H*W) -> (H*W, 批次, 通道)
        # assignment_weights: [batch_size, channels, num_codes]
        # 分配权重: [批次大小, 通道数, 码本数量]
        assignment_weights = nn.softmax(  # 应用softmax获取分配权重
            self.scaled_l2(x, self.codewords, self.scale), dim=2)  # 在码本维度应用softmax
        # aggregate  # 聚合
        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)  # 使用权重聚合特征
        return encoded_feat  # 返回编码特征


class EncModule(nn.Module):  # 定义编码注意力模块类
    def __init__(self, in_channels, num_codes):  # 初始化函数，参数：输入通道数、码本数量
        super(EncModule, self).__init__()  # 调用父类初始化
        self.encoding_project = nn.Conv2d(in_channels, in_channels, 1)  # 1x1卷积用于投影
        self.encoding = nn.Sequential(  # 编码序列模块
            Encoding(channels=in_channels, num_codes=num_codes),  # 编码层
            nn.BatchNorm(num_codes),  # 批归一化，对码本维度
            nn.ReLU())  # ReLU激活函数
        self.fc = nn.Sequential(  # 全连接序列模块
            nn.Linear(in_channels, in_channels), nn.Sigmoid())  # 线性层后接Sigmoid，生成通道权重

    def execute(self, x):  # 前向传播函数
        encoding_projection = self.encoding_project(x)  # 通过投影卷积
        encoding_feat = self.encoding(encoding_projection).mean(dim=1)  # 编码特征并在码本维度求平均
        batch_size, channels, _, _ = x.size()  # 获取输入张量的形状
        gamma = self.fc(encoding_feat)  # 通过全连接层生成通道注意力权重
        return x*gamma.view(batch_size, channels, 1, 1)  # 将注意力权重reshape并与输入相乘


def main():  # 主函数，用于测试
    attention_block = EncModule(64, 32)  # 创建编码注意力模块，64个输入通道，32个码本
    input = torch.rand([4, 64, 32, 32])  # 创建测试输入张量，形状为(4, 64, 32, 32)
    output = attention_block(input)  # 通过编码注意力模块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
