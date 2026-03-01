import os
from PIL import Image
import numpy as np

class HazeRemoval:
    def __init__(self, filename, omega=0.85, r=40, eps=1e-3, d_bilateral=9, sigma_color=75, sigma_space=75):
        self.filename = filename
        self.omega = omega
        self.r = r
        self.eps = eps
        self.d_bilateral = d_bilateral   # 双边滤波邻域直径
        self.sigma_color = sigma_color   # 颜色空间标准差
        self.sigma_space = sigma_space   # 坐标空间标准差
        self.t = 0.1                     # 最小透射率阈值

    def _ind2sub(self, array_shape, ind):
        # 将一维索引转换为二维坐标
        rows = (ind.astype('int') // array_shape[1])
        cols = (ind.astype('int') % array_shape[1])
        return rows, cols

    def _rgb2gray(self, rgb):
        # 将RGB图像转换为灰度图，使用常见加权系数
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def _box_filter(self, img, r):
        # 基于积分图的滑动窗口均值滤波（box filter）
        h, w = img.shape
        # 计算积分图，多填充一行一列
        integral = np.pad(img.cumsum(axis=0).cumsum(axis=1), ((1,0),(1,0)), mode='constant', constant_values=0)
        # 构建输出
        out = np.zeros_like(img)
        # 窗口大小
        kernel_size = (2*r + 1)
        for i in range(h):
            for j in range(w):
                # 窗口在积分图中的左上和右下坐标
                y1 = max(i - r, 0)
                x1 = max(j - r, 0)
                y2 = min(i + r, h - 1)
                x2 = min(j + r, w - 1)
                # 积分图坐标要加1
                sum_val = integral[y2+1, x2+1] - integral[y1, x2+1] - integral[y2+1, x1] + integral[y1, x1]
                area = (y2 - y1 + 1) * (x2 - x1 + 1)
                out[i, j] = sum_val / area
        return out

    def guided_filter(self, I, p, r, eps):
        """
        手动实现导向滤波。
        I: 引导图像（灰度图），p: 过滤输入，r: 窗口半径，eps: 正则项
        """
        # 1. 计算局部均值
        mean_I = self._box_filter(I, r)
        mean_p = self._box_filter(p, r)
        mean_Ip = self._box_filter(I * p, r)
        # 2. 计算协方差和方差
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = self._box_filter(I * I, r)
        var_I = mean_II - mean_I * mean_I
        # 3. 线性模型系数 a 和 b
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        # 4. 计算 a 和 b 的局部均值
        mean_a = self._box_filter(a, r)
        mean_b = self._box_filter(b, r)
        # 5. 输出
        q = mean_a * I + mean_b
        return q

    def bilateral_filter(self, img, d, sigma_color, sigma_space):
        """
        手动实现双边滤波。
        img: 单通道图像，d: 邻域直径，sigma_color: 颜色空间标准差，sigma_space: 坐标空间标准差
        """
        h, w = img.shape
        output = np.zeros_like(img)
        r = d // 2
        # 创建空间权重表
        x, y = np.mgrid[-r:r+1, -r:r+1]
        spatial_weights = np.exp(-(x**2 + y**2) / (2 * sigma_space**2))
        for i in range(h):
            for j in range(w):
                i1 = max(i - r, 0)
                i2 = min(i + r, h - 1)
                j1 = max(j - r, 0)
                j2 = min(j + r, w - 1)
                # 提取局部区域
                region = img[i1:i2+1, j1:j2+1]
                # 颜色权重
                intensity_diff = region - img[i, j]
                color_weights = np.exp(-(intensity_diff**2) / (2 * sigma_color**2))
                # 空间权重要截取对应大小
                sw = spatial_weights[(i1 - i + r):(i2 - i + r + 1), (j1 - j + r):(j2 - j + r + 1)]
                # 复合权重
                weights = sw * color_weights
                # 正规化
                W = weights.sum()
                output[i, j] = (weights * region).sum() / W
        return output

    def haze_removal(self):
        # 读取并归一化原图
        oriImage = np.array(Image.open(self.filename).convert('RGB'), dtype=np.float64) / 255.0
        gray = self._rgb2gray(oriImage).astype(np.float32)
        dark = oriImage.min(axis=2)
        # 估计大气光 A
        ind = np.unravel_index(np.argmax(dark), dark.shape)
        A = oriImage[ind[0], ind[1], :].mean()
        # 粗透射率估计
        raw_t = 1 - self.omega * dark / A
        raw_t = raw_t.astype(np.float32)
        # 导向滤波精炼透射率
        t_guided = self.guided_filter(gray, raw_t, self.r, self.eps)
        # 双边滤波进一步平滑
        t_bilateral = self.bilateral_filter(t_guided, self.d_bilateral, self.sigma_color, self.sigma_space)
        # 限制透射率
        t_bilateral[t_bilateral < self.t] = self.t
        # 复原图像
        J = np.zeros_like(oriImage)
        for c in range(3):
            J[:, :, c] = (oriImage[:, :, c] - A) / t_bilateral + A
        J = np.clip(J, 0, 1)
        # 转换回PIL Image
        return Image.fromarray((J * 255).astype(np.uint8))

if __name__ == '__main__':
    input_path = r"0151.jpg"
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(os.path.dirname(input_path), f"{name}_DCP_manual.jpg")

    haze = HazeRemoval(input_path)
    result = haze.haze_removal()
    result.save(output_path)
    result.show()
