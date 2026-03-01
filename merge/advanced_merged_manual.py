from PIL import Image
import numpy as np
from skimage import feature, exposure


def gaussian_kernel1d(sigma, truncate=4.0):
    """
    生成一维高斯核。
    sigma: 高斯标准差
    truncate: 截断半径倍数，用于控制核大小
    返回长度为(2*radius+1)的归一化核数组。
    """
    # 根据truncate计算半径
    radius = int(truncate * sigma + 0.5)
    # 生成坐标序列
    x = np.arange(-radius, radius + 1)
    # 计算高斯函数值
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    # 归一化
    kernel /= kernel.sum()
    return kernel


def gaussian_filter_manual(image, sigma):
    """
    手动实现高斯滤波（分离卷积）：
    image: 2D或3D numpy数组，最后一维为通道
    sigma: 标准差，可为标量或与空间维度匹配的序列
    返回同尺寸的滤波后图像（float类型）。
    """
    # 仅支持标量sigma
    kernel = gaussian_kernel1d(sigma)
    # 定义一维卷积函数
    def convolve1d(arr, kernel, axis):
        # np.apply_along_axis对指定轴进行一维卷积，mode='same'保持尺寸
        return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis, arr)

    # 如果是多通道图像，逐通道处理
    if image.ndim == 3:
        out = np.zeros_like(image, dtype=float)
        for c in range(image.shape[2]):
            channel = image[:, :, c]
            # 先沿垂直方向卷积，再沿水平方向卷积
            channel = convolve1d(channel, kernel, axis=0)
            channel = convolve1d(channel, kernel, axis=1)
            out[:, :, c] = channel
    else:
        # 单通道图像
        out = convolve1d(image, kernel, axis=0)
        out = convolve1d(out, kernel, axis=1)
    return out


def advanced_merge(sky_path, dcp_path, mask_path, sigma=7, canny_sigma=1.0, match_hist=True):
    # 读取图像并转为numpy数组
    sky = np.array(Image.open(sky_path))
    dcp = np.array(Image.open(dcp_path))
    # mask转为灰度
    mask = np.array(Image.open(mask_path).convert('L'))

    # 验证尺寸一致性
    if sky.shape != dcp.shape or sky.shape[:2] != mask.shape:
        raise ValueError("所有图片尺寸必须一致")

    # 二值化mask
    mask_bin = np.where(mask >= 128, 255, 0).astype(np.uint8)

    # 色彩直方图匹配（可选）
    if match_hist:
        dcp = exposure.match_histograms(dcp, sky, channel_axis=-1)

    # 构建alpha通道（浮点0-1）
    alpha = mask_bin.astype(float) / 255.0

    # 边缘检测
    edges = feature.canny(alpha, sigma=canny_sigma).astype(float)
    # 使用手动高斯滤波平滑边缘，并缩放影响强度
    edge_smooth = gaussian_filter_manual(edges, sigma=2)
    edge_mask = edge_smooth * 0.8

    # 对alpha进行手动高斯平滑
    base_alpha = gaussian_filter_manual(alpha, sigma)
    # 减去边缘影响并裁剪到[0,1]
    final_alpha = np.clip(base_alpha - edge_mask, 0, 1)

    # 扩展到3通道
    final_alpha_3d = final_alpha[:, :, np.newaxis]

    # 图像融合：sky * alpha + dcp * (1-alpha)
    blended = sky * final_alpha_3d + dcp * (1 - final_alpha_3d)

    return Image.fromarray(blended.astype(np.uint8))


if __name__ == '__main__':
    result = advanced_merge('0151_sky.jpg', '0151_DCP.jpg', '0151_mask.jpg', sigma=7, canny_sigma=1.5)
    result.save('0151_merged.jpg')
