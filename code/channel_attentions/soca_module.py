# Second-Order Attention Network for Single Image Super-Resolution (CVPR 2019)
# 单图像超分辨率的二阶注意力网络 (CVPR 2019)
import torch  # 导入PyTorch库
from torch import nn, Function  # 导入神经网络模块和Function类


class Covpool(Function):  # 定义协方差池化函数类
    def execute(self, input):  # 前向传播函数
        x = input  # 输入张量
        batchSize = x.data.shape[0]  # 获取批次大小
        dim = x.data.shape[1]  # 获取通道数
        h = x.data.shape[2]  # 获取高度
        w = x.data.shape[3]  # 获取宽度
        M = h*w  # 计算空间维度总数
        x = x.reshape(batchSize, dim, M)  # 重塑为(批次, 通道, H*W)
        I_hat = (-1./M/M)*torch.ones((M, M)) + (1./M)*torch.eye((M, M))  # 计算中心化矩阵
        I_hat = I_hat.view(1, M, M).repeat(batchSize, 1, 1)  # 扩展到批次维度
        y = nn.bmm(nn.bmm(x, I_hat), x.transpose(0, 2, 1))  # 计算协方差矩阵：x * I_hat * x^T
        self.save_vars = (input, I_hat)  # 保存变量用于反向传播
        return y  # 返回协方差矩阵

    def grad(self, grad_output):  # 反向传播函数
        input, I_hat = self.save_vars  # 获取保存的变量
        x = input  # 输入张量
        batchSize = x.data.shape[0]  # 获取批次大小
        dim = x.data.shape[1]  # 获取通道数
        h = x.data.shape[2]  # 获取高度
        w = x.data.shape[3]  # 获取宽度
        M = h*w  # 计算空间维度总数
        x = x.reshape(batchSize, dim, M)  # 重塑为(批次, 通道, H*W)
        grad_input = grad_output + grad_output.transpose(0, 2, 1)  # 对称化梯度
        grad_input = nn.bmm(nn.bmm(grad_input, x), I_hat)  # 计算输入梯度
        grad_input = grad_input.reshape(batchSize, dim, h, w)  # 恢复原始形状
        return grad_input  # 返回输入梯度


class Sqrtm(Function):  # 定义矩阵平方根函数类
    def execute(self, input, iterN):  # 前向传播函数，iterN为迭代次数
        x = input  # 输入张量
        batchSize = x.data.shape[0]  # 获取批次大小
        dim = x.data.shape[1]  # 获取维度
        I3 = 3.0*torch.eye((dim, dim)).view(1,  # 创建3倍单位矩阵
                                              dim, dim).repeat(batchSize, 1, 1)
        normA = (1.0/3.0)*x.matmul(I3).sum(dim=1).sum(dim=1)  # 计算矩阵的谱范数
        A = x / (normA.view(batchSize, 1, 1).expand_as(x))  # 归一化矩阵
        Y = torch.zeros((batchSize, iterN, dim, dim))  # 初始化Y序列
        Y.requires_grad = False  # 不需要梯度
        Z = torch.eye((dim, dim)).view(  # 初始化Z序列
            1, dim, dim).repeat(batchSize, iterN, 1, 1)
        if iterN < 2:  # 如果迭代次数小于2
            ZY = 0.5*(I3 - A)  # 计算ZY
            Y[:, 0, :, :] = nn.bmm(A, ZY)  # 更新Y
        else:  # 如果迭代次数大于等于2
            ZY = 0.5*(I3 - A)  # 计算初始ZY
            Y[:, 0, :, :] = nn.bmm(A, ZY)  # 初始化Y[0]
            Z[:, 0, :, :] = ZY  # 初始化Z[0]
            for i in range(1, iterN-1):  # 迭代计算
                ZY = 0.5*nn.bmm(I3 - Z[:, i-1, :, :], Y[:, i-1, :, :])  # 更新ZY
                Y[:, i, :, :] = nn.bmm(Y[:, i-1, :, :], ZY)  # 更新Y[i]
                Z[:, i, :, :] = nn.bmm(ZY, Z[:, i-1, :, :])  # 更新Z[i]
            ZY = nn.bmm(  # 计算最终的ZY
                nn.bmm(0.5*Y[:, iterN-2, :, :], I3 - Z[:, iterN-2, :, :]), Y[:, iterN-2, :, :])
        y = ZY*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)  # 计算最终结果
        self.save_vars = (input, A, ZY, normA, Y, Z)  # 保存变量用于反向传播
        self.iterN = iterN  # 保存迭代次数
        return y  # 返回矩阵平方根

    def grad(self, grad_output):  # 反向传播函数
        input, A, ZY, normA, Y, Z = self.save_vars  # 获取保存的变量
        iterN = self.iterN  # 获取迭代次数
        x = input  # 输入张量
        batchSize = x.data.shape[0]  # 获取批次大小
        dim = x.data.shape[1]  # 获取维度
        der_postCom = grad_output * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)  # 计算后处理梯度
        der_postComAux = (grad_output*ZY).sum(dim=1).sum(dim=1) / (2*torch.sqrt(normA))  # 计算辅助梯度
        I3 = 3.0*torch.eye((dim, dim)).view(1, dim, dim).repeat(batchSize, 1, 1)  # 创建3倍单位矩阵
        if iterN < 2:  # 如果迭代次数小于2
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_postCom))  # 计算梯度
        else:  # 如果迭代次数大于等于2
            dldY = 0.5*(nn.bmm(der_postCom, I3 - nn.bmm(Y[:, iterN-2, :, :], Z[:, iterN-2, :, :])) - nn.bmm(  # 计算对Y的梯度
                nn.bmm(Z[:, iterN-2, :, :], Y[:, iterN-2, :, :]), der_postCom))
            dldZ = -0.5*nn.bmm(  # 计算对Z的梯度
                nn.bmm(Y[:, iterN-2, :, :], der_postCom), Y[:, iterN-2, :, :])
            for i in range(iterN-3, -1, -1):  # 反向迭代计算梯度
                YZ = I3 - nn.bmm(Y[:, i, :, :], Z[:, i, :, :])  # 计算YZ
                ZY = nn.bmm(Z[:, i, :, :], Y[:, i, :, :])  # 计算ZY
                dldY_ = 0.5*(nn.bmm(dldY, YZ) -  # 更新对Y的梯度
                             nn.bmm(nn.bmm(Z[:, i, :, :], dldZ), Z[:, i, :, :]) -
                             nn.bmm(ZY, dldY))
                dldZ_ = 0.5*(nn.bmm(YZ, dldZ) -  # 更新对Z的梯度
                             nn.bmm(nn.bmm(Y[:, i, :, :], dldY), Y[:, i, :, :]) -
                             nn.bmm(dldZ, ZY))
                dldY = dldY_  # 更新dldY
                dldZ = dldZ_  # 更新dldZ
            der_NSiter = 0.5*(nn.bmm(dldY, I3 - A) - dldZ - nn.bmm(A, dldY))  # 计算最终梯度
        grad_input = der_NSiter / (normA.view(batchSize, 1, 1).expand_as(x))  # 计算输入梯度
        grad_aux = der_NSiter.matmul(x).sum(dim=1).sum(dim=1)  # 计算辅助梯度
        for i in range(batchSize):  # 遍历每个批次
            grad_input[i, :, :] += (der_postComAux[i]  # 添加辅助梯度项
                                    - grad_aux[i] / (normA[i] * normA[i])) * torch.ones((dim,)).diag()
        return grad_input, None  # 返回输入梯度


class SOCA(nn.Module):  # 定义SOCA（Second-Order Channel Attention）模块类
    def __init__(self, channel, reduction=8):  # 初始化函数，参数：通道数、降维比例（默认8）
        super().__init__()  # 调用父类初始化

        self.conv_du = nn.Sequential(  # 定义卷积序列，用于生成注意力权重
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),  # 降维卷积
            nn.ReLU(),  # ReLU激活函数
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),  # 恢复维度的卷积
            nn.Sigmoid()  # Sigmoid激活函数，生成0-1之间的权重
        )
        self.CovpoolLayer = Covpool()  # 创建协方差池化层
        self.SqrtmLayer = Sqrtm()  # 创建矩阵平方根层

    def execute(self, x):  # 前向传播函数
        b, c, h, w = x.shape  # 获取输入张量的形状

        h1 = 1000  # 设定最大高度阈值
        w1 = 1000  # 设定最大宽度阈值
        if h < h1 and w < w1:  # 如果高度和宽度都小于阈值
            x_sub = x  # 直接使用原始输入
        elif h < h1 and w > w1:  # 如果高度小于阈值但宽度大于阈值
            W = (w - w1) // 2  # 计算裁剪起始位置
            x_sub = x[:, :, :, W:(W + w1)]  # 在宽度方向裁剪
        elif w < w1 and h > h1:  # 如果宽度小于阈值但高度大于阈值
            H = (h - h1) // 2  # 计算裁剪起始位置
            x_sub = x[:, :, H:H + h1, :]  # 在高度方向裁剪
        else:  # 如果高度和宽度都大于阈值
            H = (h - h1) // 2  # 计算高度裁剪起始位置
            W = (w - w1) // 2  # 计算宽度裁剪起始位置
            x_sub = x[:, :, H:(H + h1), W:(W + w1)]  # 在两个方向都裁剪

        # MPN-COV  # 矩阵幂归一化协方差
        cov_mat = self.CovpoolLayer(x_sub)  # 计算协方差矩阵
        cov_mat_sqrt = self.SqrtmLayer(cov_mat, 5)  # 计算协方差矩阵的平方根，迭代5次

        cov_mat_sum = torch.mean(cov_mat_sqrt, 1)  # 在通道维度求平均
        cov_mat_sum = cov_mat_sum.view(b, c, 1, 1)  # 重塑为(批次, 通道, 1, 1)

        y_cov = self.conv_du(cov_mat_sum)  # 通过卷积序列生成注意力权重

        return y_cov*x  # 将注意力权重与输入相乘


def main():  # 主函数，用于测试
    attention_block = SOCA(64)  # 创建SOCA模块，64个通道
    input = torch.rand([4, 64, 32, 32])  # 创建测试输入张量，形状为(4, 64, 32, 32)
    output = attention_block(input)  # 通过SOCA模块进行前向传播
    torch.autograd.grad(output, input)  # 计算梯度（测试反向传播）
    print(input.size(), output.size())  # 打印输入和输出的尺寸


if __name__ == '__main__':  # 如果是主程序运行
    main()  # 调用主函数
