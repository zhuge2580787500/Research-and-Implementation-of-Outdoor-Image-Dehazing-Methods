from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import feature, exposure

def advanced_merge(sky_path, dcp_path, mask_path, sigma=7, canny_sigma=1.0, match_hist=True):
    # 读取图像
    sky = np.array(Image.open(sky_path))
    dcp = np.array(Image.open(dcp_path))
    mask = np.array(Image.open(mask_path).convert('L'))

    # 验证尺寸一致性
    if sky.shape != dcp.shape or sky.shape[:2] != mask.shape:
        raise ValueError("所有图片尺寸必须一致")

    # 二值化处理
    mask_bin = np.where(mask >= 128, 255, 0).astype(np.uint8)
    
    # 色彩一致性处理（核心改进1）
    if match_hist:
        dcp = exposure.match_histograms(dcp, sky, channel_axis=-1)

    # 创建浮点型alpha通道
    alpha = mask_bin.astype(float)/255.0
    
    # 边缘检测优化（核心改进2）
    edges = feature.canny(alpha, sigma=canny_sigma).astype(float)
    edge_mask = gaussian_filter(edges, sigma=2) * 0.8  # 创建边缘影响区域
    
    # 双重alpha混合
    base_alpha = gaussian_filter(alpha, sigma=sigma)
    final_alpha = np.clip(base_alpha - edge_mask, 0, 1)
    
    # 三维化处理
    final_alpha_3d = final_alpha[:, :, np.newaxis]
    
    # 图像混合
    blended = sky * final_alpha_3d + dcp * (1 - final_alpha_3d)
    
    return Image.fromarray(blended.astype(np.uint8))

if __name__ == '__main__':
    result = advanced_merge('0151_sky.jpg', '0151_DCP.jpg', '0151_mask.jpg', 
                        sigma=7, canny_sigma=1.5)
    result.save('0151_merged.jpg')