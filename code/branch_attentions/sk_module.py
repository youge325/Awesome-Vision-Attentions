import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块


class SKModule(nn.Module):  # 定义SK（Selective Kernel）模块类
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):  # 初始化函数，包含各种参数
        """ Constructor  构造函数
        Args:  参数：
            features: input channel dimensionality.  输入通道维度
            M: the number of branchs.  分支数量
            G: num of convolution groups.  卷积分组数
            r: the ratio for compute d, the length of z.  计算d的比例，z向量的长度
            stride: stride, default 1.  步长，默认为1
            L: the minimum dim of the vector z in paper, default 32.  论文中向量z的最小维度，默认为32
        """
        super(SKModule, self).__init__()  # 调用父类初始化
        d = max(int(features/r), L)  # 计算中间层维度d，取features/r和L的最大值
        self.M = M  # 存储分支数量
        self.features = features  # 存储特征通道数
        self.convs = nn.ModuleList([])  # 创建卷积分支的模块列表
        for i in range(M):  # 遍历每个分支
            self.convs.append(nn.Sequential(  # 为每个分支添加卷积块
                nn.Conv2d(features, features, kernel_size=3, stride=stride,  # 3x3卷积层
                          padding=1+i, dilation=1+i, groups=G, bias=False),  # 填充和膨胀系数随分支递增
                nn.BatchNorm2d(features),  # 批归一化层
                nn.ReLU()  # ReLU激活函数
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 全局自适应平均池化，输出1x1
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),  # 降维的1x1卷积
                                nn.BatchNorm2d(d),  # 批归一化
                                nn.ReLU())  # ReLU激活
        self.fcs = nn.ModuleList([])  # 创建分支特定的全连接层列表
        for i in range(M):  # 为每个分支创建一个全连接层
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)  # 1x1卷积，将d维特征映射回features维
            )
        self.softmax = nn.Softmax(dim=1)  # Softmax层，用于生成注意力权重

    def execute(self, x):  # 前向传播函数

        batch_size = x.shape[0]  # 获取批次大小

        feats = [conv(x) for conv in self.convs]  # 通过每个卷积分支处理输入，得到M个特征图
        feats = torch.concat(feats, dim=1)  # 在通道维度上连接所有分支的特征图
        feats = feats.view(batch_size, self.M, self.features,  # 重塑张量为(batch_size, M, features, H, W)
                           feats.shape[2], feats.shape[3])

        feats_U = torch.sum(feats, dim=1)  # 对所有分支特征求和，得到融合特征U
        feats_S = self.gap(feats_U)  # 对融合特征进行全局平均池化，得到全局描述符S
        feats_Z = self.fc(feats_S)  # 通过全连接层处理S，得到紧凑特征Z

        attention_vectors = [fc(feats_Z) for fc in self.fcs]  # 为每个分支生成注意力向量
        attention_vectors = torch.concat(attention_vectors, dim=1)  # 在通道维度上连接所有注意力向量
        attention_vectors = attention_vectors.view(  # 重塑注意力向量为(batch_size, M, features, 1, 1)
            batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)  # 应用softmax生成归一化的注意力权重

        feats_V = torch.sum(feats*attention_vectors, dim=1)  # 使用注意力权重对分支特征进行加权融合

        return feats_V  # 返回最终的融合特征


def main():  # 主函数，用于测试
    attention_block = SKModule(64)  # 创建一个SK模块，特征通道数为64
    input = torch.rand([4, 64, 32, 32])  # 创建测试输入张量，形状为(4, 64, 32, 32)
    output = attention_block(input)  # 通过SK模块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
