# DIANet: Dense-and-Implicit Attention Network (AAAI 2020)
# DIANet：密集隐式注意力网络 (AAAI 2020)
import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块


class small_cell(nn.Module):  # 定义小型单元类
    def __init__(self, input_size, hidden_size):  # 初始化函数，参数：输入尺寸、隐藏层尺寸
        """"Constructor of the class"""  # 类的构造函数
        super(small_cell, self).__init__()  # 调用父类初始化
        self.seq = nn.Sequential(nn.Linear(input_size, input_size // 4),  # 创建序列模块：第一个线性层，维度减少到1/4
                                 nn.ReLU(),  # ReLU激活函数
                                 nn.Linear(input_size // 4, 4 * hidden_size))  # 第二个线性层，扩展到4倍隐藏层尺寸

    def execute(self, x):  # 前向传播函数
        return self.seq(x)  # 通过序列模块处理输入并返回


class LSTMCell(nn.Module):  # 定义LSTM单元类
    def __init__(self, input_size, hidden_size, nlayers, dropout=0.1):  # 初始化函数，参数：输入尺寸、隐藏层尺寸、层数、dropout率
        """"Constructor of the class"""  # 类的构造函数
        super(LSTMCell, self).__init__()  # 调用父类初始化

        self.nlayers = nlayers  # 存储层数
        self.dropout = nn.Dropout(p=dropout)  # 创建dropout层

        ih, hh = [], []  # 初始化输入到隐藏层和隐藏层到隐藏层的权重列表
        for i in range(nlayers):  # 遍历每一层
            if i == 0:  # 如果是第一层
                # ih.append(nn.Linear(input_size, 4 * hidden_size))  # 注释掉的标准线性层
                ih.append(small_cell(input_size, hidden_size))  # 使用小型单元处理输入到隐藏层的变换
                # hh.append(nn.Linear(hidden_size, 4 * hidden_size))  # 注释掉的标准线性层
                hh.append(small_cell(hidden_size, hidden_size))  # 使用小型单元处理隐藏层到隐藏层的变换
            else:  # 如果不是第一层
                ih.append(nn.Linear(hidden_size, 4 * hidden_size))  # 标准线性层：隐藏层到4倍隐藏层
                hh.append(nn.Linear(hidden_size, 4 * hidden_size))  # 标准线性层：隐藏层到4倍隐藏层
        self.w_ih = nn.ModuleList(ih)  # 将输入到隐藏层权重转换为模块列表
        self.w_hh = nn.ModuleList(hh)  # 将隐藏层到隐藏层权重转换为模块列表

    def execute(self, input, hidden):  # 前向传播函数，参数：输入、隐藏状态
        """"Defines the forward computation of the LSTMCell"""  # 定义LSTM单元的前向计算
        hy, cy = [], []  # 初始化输出隐藏状态和细胞状态列表
        for i in range(self.nlayers):  # 遍历每一层
            hx, cx = hidden[0][i], hidden[1][i]  # 获取当前层的隐藏状态和细胞状态
            gates = self.w_ih[i](input) + self.w_hh[i](hx)  # 计算门控值：输入门、遗忘门、候选值、输出门
            i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)  # 将门控值分成4部分
            i_gate = i_gate.sigmoid()  # 输入门使用sigmoid激活
            f_gate = f_gate.sigmoid()  # 遗忘门使用sigmoid激活
            c_gate = torch.tanh(c_gate)  # 候选值使用tanh激活
            o_gate = o_gate.sigmoid()  # 输出门使用sigmoid激活
            ncx = (f_gate * cx) + (i_gate * c_gate)  # 计算新的细胞状态：遗忘门*旧细胞状态 + 输入门*候选值
            # nhx = o_gate * torch.tanh(ncx)  # 注释掉的标准隐藏状态计算
            nhx = o_gate * ncx.sigmoid()  # 计算新的隐藏状态：输出门 * sigmoid(新细胞状态)
            cy.append(ncx)  # 将新细胞状态添加到列表
            hy.append(nhx)  # 将新隐藏状态添加到列表
            input = self.dropout(nhx)  # 对隐藏状态应用dropout，作为下一层的输入

        hy, cy = torch.stack(hy, 0), torch.stack(  # 将隐藏状态和细胞状态堆叠成张量
            cy, 0)  # number of layer * batch * hidden  # 形状：层数 * 批次 * 隐藏维度
        return hy, cy  # 返回隐藏状态和细胞状态


class Attention(nn.Module):  # 定义注意力模块类
    def __init__(self, channel):  # 初始化函数，参数：通道数
        super(Attention, self).__init__()  # 调用父类初始化
        self.lstm = LSTMCell(channel, channel, 1)  # 创建单层LSTM单元

        self.GlobalAvg = nn.AdaptiveAvgPool2d((1, 1))  # 全局自适应平均池化，输出1x1
        self.relu = nn.ReLU()  # ReLU激活函数

    def execute(self, x):  # 前向传播函数
        org = x  # 保存原始输入用于残差连接
        seq = self.GlobalAvg(x)  # 全局平均池化，得到每个通道的全局特征
        seq = seq.view(seq.size(0), seq.size(1))  # 将特征图reshape为(batch_size, channel)
        ht = torch.zeros((1, seq.size(0), seq.size(  # 初始化隐藏状态张量
            1)))  # 1 mean number of layers  # 1表示层数
        ct = torch.zeros((1, seq.size(0), seq.size(1)))  # 初始化细胞状态张量
        ht, ct = self.lstm(seq, (ht, ct))  # 通过LSTM处理序列，得到隐藏状态和细胞状态  # 1 * batch size * length
        # ht = self.sigmoid(ht)  # 注释掉的sigmoid激活
        x = x * (ht[-1].view(ht.size(1), ht.size(2), 1, 1))  # 使用LSTM输出的注意力权重对原始特征进行加权
        x += org  # 添加残差连接
        x = self.relu(x)  # 应用ReLU激活函数

        return x  # , list  # 返回处理后的特征


def main():  # 主函数，用于测试
    attention_block = Attention(64)  # 创建注意力模块，通道数为64
    input = torch.rand([4, 64, 32, 32])  # 创建测试输入张量，形状为(4, 64, 32, 32)
    output = attention_block(input)  # 通过注意力模块进行前向传播
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
