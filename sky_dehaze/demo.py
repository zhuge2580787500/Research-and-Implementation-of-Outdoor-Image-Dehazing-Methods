import os
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from PIL import Image
from utils import logger
from config import get_config
from model import ACT

@logger
def make_test_data(cfg, img_path, device):
    # 读取原始图像并记录尺寸
    img = Image.open(str(img_path))
    original_size = img.size  # (W, H)
    
    # 动态调整尺寸（保持长宽比，适配模型结构）
    w, h = img.size
    scale = cfg.base_size / min(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 确保调整后的尺寸是16的倍数
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    
    # 创建动态变换流程
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((new_h, new_w)),
        torchvision.transforms.ToTensor()
    ])
    x = transform(img).unsqueeze(0).to(device)
    
    return x, original_size

@logger
def load_pretrain_network(cfg, device):
    # 加载模型并加载预训练参数
    net = ACT().to(device)
    ckpt_path = os.path.join(cfg.model_dir, cfg.ckpt)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    return net

def main(cfg, input_image_path, output_image_path=None):
    # 基础配置
    print(cfg)
    if cfg.gpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载网络
    network = load_pretrain_network(cfg, device)
    network.eval()

    # 处理输入数据
    test_image, original_size = make_test_data(cfg, input_image_path, device)
    
    with torch.no_grad():
        dehaze_image = network(test_image)
    
    # 恢复原始尺寸
    orig_size = original_size[::-1]  # 转换为(H, W)
    dehaze_image = torchvision.transforms.functional.resize(
        dehaze_image,
        orig_size,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    )
    
    
    # 确定输出路径
    if output_image_path is None:
        # 如果没有指定输出路径，则在输入文件同目录下生成
        dirname = os.path.dirname(input_image_path)
        filename = os.path.basename(input_image_path)
        # 分离文件名和扩展名
        name, ext = os.path.splitext(filename)
        # 构建新文件名：原文件名_sky + 原扩展名
        output_image_path = os.path.join(dirname, f"{name}_sky{ext}")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    
    # 保存结果
    torchvision.utils.save_image(dehaze_image, output_image_path)
    print(f"Processed image saved to: {output_image_path}")

if __name__ == '__main__':

    config_args, _ = get_config()
    config_args.base_size = 480  # 可调整的基础尺寸
    config_args.use_gpu = True
    config_args.gpu = 0 
    config_args.model_dir = r'C:\Users\lenovo\Desktop\bishe_final\sky_dehaze\model'
    config_args.ckpt = 'dehaze_chromatic_100.pkl'
   

    input_image_path = r"C:\Users\lenovo\Desktop\bishe_final\0151.jpg"  # 替换为输入图片路径
    output_image_path = "output_image.jpg"  
    
    # 调用方式1：不指定输出路径，默认在输入文件同目录生成
    main(config_args, input_image_path)
    
    # 调用方式2：指定输出路径
    # main(config_args, input_image_path, output_image_path)