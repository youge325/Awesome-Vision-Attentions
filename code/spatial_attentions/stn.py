import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块
import numpy as np  # 导入NumPy库


def get_pixel_value(img, x, y):  # 获取像素值函数
    B, C, H, W = img.shape  # 获取图像形状
    return img.reindex([B, C, H, W], ['i0', 'i1', '@e0(i0, i2, i3)','@e1(i0, i2, i3)'], extras=[x, y])  # 根据坐标重新索引获取像素值


def affine_grid_generator(height, width, theta):  # 仿射网格生成器函数
    num_batch = theta.shape[0]  # 获取批次大小
    
    # create normalized 2D grid
    x = torch.linspace(-1.0, 1.0, width)  # 生成宽度方向的归一化坐标
    y = torch.linspace(-1.0, 1.0, height)  # 生成高度方向的归一化坐标
    x_t, y_t = torch.meshgrid(x, y)  # 创建网格坐标

    # flatten
    x_t_flat = x_t.reshape(-1)  # 展平x坐标
    y_t_flat = y_t.reshape(-1)  # 展平y坐标
    print(x_t.shape)  # 打印网格形状
    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = torch.ones_like(x_t_flat)  # 创建全1向量用于齐次坐标
    sampling_grid = torch.stack([x_t_flat, y_t_flat, ones])  # 堆叠为齐次坐标形式

    # repeat grid num_batch times
    sampling_grid = sampling_grid.unsqueeze(0).expand(num_batch, -1, -1)  # 扩展到批次维度

    # transform the sampling grid - batch multiply
    batch_grids = torch.matmul(theta, sampling_grid)  # 仿射变换：theta矩阵乘以采样网格

    # reshape to (num_batch, H, W, 2)
    batch_grids = batch_grids.reshape(num_batch, 2, height, width)  # 重塑为(批次, 2, 高度, 宽度)
    return batch_grids  # 返回变换后的网格


def bilinear_sampler(img, x, y):  # 双线性采样函数
    B, C, H ,W = img.shape  # 获取图像形状
    max_y = H - 1  # 最大y坐标
    max_x = W - 1  # 最大x坐标

    # rescale x and y to [0, W-1/H-1]
    x = 0.5 * (x + 1.0) * (max_x-1)  # 将x从[-1,1]重缩放到[0, W-1]
    y = 0.5 * (y + 1.0) * (max_y-1)  # 将y从[-1,1]重缩放到[0, H-1]

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = torch.floor(x).to(torch.int32)  # 左上角x坐标（向下取整）
    x1 = x0 + 1  # 右上角x坐标
    y0 = torch.floor(y).to(torch.int32)  # 左上角y坐标（向下取整）
    y1 = y0 + 1  # 左下角y坐标

    x0 = torch.minimum(torch.maximum(0, x0), max_x)  # 限制x0在有效范围内
    x1 = torch.minimum(torch.maximum(0, x1), max_x)  # 限制x1在有效范围内
    y0 = torch.minimum(torch.maximum(0, y0), max_y)  # 限制y0在有效范围内
    y1 = torch.minimum(torch.maximum(0, y1), max_y)  # 限制y1在有效范围内

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)  # 获取左上角像素值
    Ib = get_pixel_value(img, x0, y1)  # 获取左下角像素值
    Ic = get_pixel_value(img, x1, y0)  # 获取右上角像素值
    Id = get_pixel_value(img, x1, y1)  # 获取右下角像素值

    # calculate deltas
    wa = (x1-x) * (y1-y)  # 左上角权重
    wb = (x1-x) * (y-y0)  # 左下角权重
    wc = (x-x0) * (y1-y)  # 右上角权重
    wd = (x-x0) * (y-y0)  # 右下角权重

    # compute output
    out = wa*Ia + wb*Ib + wc*Ic + wd*Id  # 双线性插值：加权组合四个角的像素值
    return out  # 返回插值结果

class STN(nn.Module):  # 定义空间变换网络类
    def __init__(self):  # 初始化函数
        super(STN, self).__init__()  # 调用父类初始化
    
    def execute(self, x1, theta):  # 前向传播函数，参数：输入图像x1，变换参数theta
        B, C, H, W = x1.shape  # 获取输入图像形状
        theta = theta.reshape(-1, 2, 3)  # 将theta重塑为(批次, 2, 3)的仿射变换矩阵
        
        batch_grids = affine_grid_generator(H, W, theta)  # 生成仿射变换网格
        
        x_s = batch_grids[:, 0, :, :]  # 提取变换后的x坐标
        y_s = batch_grids[:, 1, :, :]  # 提取变换后的y坐标

        out_fmap = bilinear_sampler(x1, x_s, y_s)  # 使用双线性采样进行空间变换

        return out_fmap  # 返回变换后的特征图


def main():  # 主函数，用于测试
    stn = STN()  # 创建空间变换网络
    x = torch.randn(1, 3, 224, 224)  # 创建随机输入图像
    theta = torch.tensor(np.random.uniform(0,1,(1,6)))  # 创建随机仿射变换参数
    y = stn(x, theta)  # 通过空间变换网络进行变换
    print(y)  # 打印输出

if __name__ == "__main__":  # 如果是主程序运行
    main()  # 调用主函数
