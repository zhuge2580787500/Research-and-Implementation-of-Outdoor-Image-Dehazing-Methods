import cv2
import numpy as np
import os


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 10 * np.log10(max_pixel ** 2 / mse)
    return psnr


# def compare_images_with_reference(reference_image_name):
#     # 读取参考图像
#     ref_img = cv2.imread(reference_image_name, cv2.IMREAD_COLOR)
#     if ref_img is None:
#         print(f"无法读取参考图像 {reference_image_name}。")
#         return

#     # 获取当前文件夹中所有图像文件
#     image_extensions = ['.jpg', '.jpeg', '.png']
#     all_images = [f for f in os.listdir('.') if any(f.lower().endswith(ext) for ext in image_extensions)]

#     # 遍历所有图像，计算PSNR
#     for image_name in all_images:
#         if image_name != reference_image_name:
#             # 读取当前图像
#             current_img = cv2.imread(image_name, cv2.IMREAD_COLOR)
#             if current_img is not None and current_img.shape == ref_img.shape:
#                 # 计算PSNR
#                 psnr = calculate_psnr(ref_img, current_img)
#                 print(f"与 {image_name} 的PSNR: {psnr:.2f} dB")
#             else:
#                 print(f"无法处理 {image_name}，可能是图像读取失败或尺寸不匹配。")

def compare_images_with_reference(reference_image_name, target_image_path=None):
    psnr_results = []  # 初始化PSNR结果列表
    
    # 读取参考图像（彩色模式）
    ref_img = cv2.imread(reference_image_name, cv2.IMREAD_COLOR)
    if ref_img is None:
        print(f"错误：无法读取参考图像 {reference_image_name}")
        return psnr_results  # 返回空列表
    
    # 处理目标图像指定逻辑
    if target_image_path is not None:
        # 检查目标图像是否存在且为支持的格式
        if not os.path.exists(target_image_path):
            print(f"错误：目标图像路径 {target_image_path} 不存在")
            return psnr_results
        
        if not any(target_image_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            print(f"错误：目标图像 {target_image_path} 不是支持的格式（仅支持 jpg/jpeg/png）")
            return psnr_results
        
        # 读取目标图像并比较
        current_img = cv2.imread(target_image_path, cv2.IMREAD_COLOR)
        if current_img is None:
            print(f"错误：无法读取目标图像 {target_image_path}")
            return psnr_results
        
        if current_img.shape != ref_img.shape:
            print(f"错误：目标图像 {target_image_path} 与参考图像尺寸不匹配（参考尺寸：{ref_img.shape}，目标尺寸：{current_img.shape}）")
            return psnr_results
        
        # 计算 PSNR 并添加到结果列表
        psnr = calculate_psnr(ref_img, current_img)
        psnr_results.append(psnr)
        print(f"参考图像 {reference_image_name} 与目标图像 {target_image_path} 的 PSNR: {psnr:.2f} dB")
        return psnr_results
    
    # 未指定目标图像时，遍历当前文件夹所有图像比较
    image_extensions = ['.jpg', '.jpeg', '.png']
    all_images = [f for f in os.listdir('.') if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    for image_name in all_images:
        if image_name == reference_image_name:
            continue  # 跳过参考图像自身
        
        current_img = cv2.imread(image_name, cv2.IMREAD_COLOR)
        if current_img is None:
            print(f"警告：无法读取图像 {image_name}，跳过")
            continue
        
        if current_img.shape != ref_img.shape:
            print(f"警告：图像 {image_name} 与参考图像尺寸不匹配（参考尺寸：{ref_img.shape}，当前尺寸：{current_img.shape}），跳过")
            continue
        
        # 计算 PSNR 并添加到结果列表
        psnr = calculate_psnr(ref_img, current_img)
        psnr_results.append(psnr)
        print(f"参考图像 {reference_image_name} 与 {image_name} 的 PSNR: {psnr:.2f} dB")
    
    return psnr_results


def main():
    reference_image_name = '0151.png'
    compare_images_with_reference(reference_image_name)


if __name__ == "__main__":
    main()