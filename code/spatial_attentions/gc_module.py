import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块


class GlobalContextBlock(nn.Module):  # 定义全局上下文块类
    def __init__(self,  # 初始化函数
                 inplanes,  # 输入通道数
                 ratio):  # 压缩比例
        super(GlobalContextBlock, self).__init__()  # 调用父类初始化
        self.inplanes = inplanes  # 存储输入通道数
        self.ratio = ratio  # 存储压缩比例
        self.planes = int(inplanes * ratio)  # 计算压缩后的通道数
        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)  # 生成空间注意力mask的1x1卷积
        self.softmax = nn.Softmax(dim=2)  # Softmax激活函数，沿第2维应用

        self.channel_add_conv = nn.Sequential(  # 通道变换序列
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),  # 降维：输入通道->压缩通道
            nn.LayerNorm([self.planes, 1, 1]),  # 层归一化
            nn.ReLU(),  # ReLU激活函数
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1))  # 升维：压缩通道->输入通道

    def spatial_pool(self, x):  # 空间池化函数
        batch, channel, height, width = x.size()  # 获取输入张量的形状

        input_x = x  # 保存原始输入
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)  # 重塑为(N, C, H*W)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)  # 在第1维增加一个维度
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)  # 生成上下文mask
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)  # 重塑mask为(N, 1, H*W)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)  # 应用softmax得到注意力权重
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)  # 在最后一维增加一个维度
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)  # 矩阵乘法：加权聚合空间信息
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)  # 重塑为全局上下文向量

        return context  # 返回全局上下文

    def execute(self, x):  # 前向传播函数
        # [N, C, 1, 1]
        context = self.spatial_pool(x)  # 获取全局上下文

        out = x  # 保存原始输入
        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)  # 通过通道变换序列处理上下文
        out = out + channel_add_term  # 残差连接：原始输入加上通道变换后的上下文

        return out  # 返回最终输出


def main():  # 主函数，用于测试
    attention_block = GlobalContextBlock(64, 1/4)  # 创建全局上下文块，64通道，压缩比例1/4
    input = torch.rand([4, 64, 32, 32])  # 创建随机输入张量
    output = attention_block(input)  # 通过全局上下文块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
