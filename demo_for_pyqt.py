import os
import sys
import DCP_dehaze.DCP_filter
# import DCP_dehaze.DCP_filter_manual
import DCP_dehaze.DCP_filter_manual
from sky_dehaze.config import get_config
import sky_dehaze.demo2
import sky_divide.sky_divide
import merge.advanced_merged
# import merge.advanced_merged_manual
import evaluate.psnr
import evaluate.ssim
from PIL import Image, ImageDraw, ImageFont


def DCP(input_path,output_path=''):
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(os.path.dirname(input_path), f"{name}_DCP.jpg")
    result = DCP_dehaze.DCP_filter.HazeRemoval(input_path).haze_removal()
    # result = DCP_dehaze.DCP_filter_manual.HazeRemoval(input_path).haze_removal()
    result.save(output_path)
    print(output_path+" done")

def sky(input_path,output_path=''):

    model_dir = os.path.join(os.getcwd(), r"sky_dehaze\model")

    config_args, _ = get_config()
    config_args.base_size = 480  # 可调整的基础尺寸
    config_args.use_gpu = True
    config_args.gpu = 0 
    # config_args.model_dir = r'C:\Users\lenovo\Desktop\bishe_final\sky_dehaze\model'
    config_args.model_dir = model_dir
    config_args.ckpt = 'dehaze_chromatic_100.pkl'
   

    name, ext = os.path.splitext(filename)
    output_path = os.path.join(os.path.dirname(input_path), f"{name}_sky.jpg")

    
    
    # 调用方式1：不指定输出路径，默认在输入文件同目录生成
    # sky_dehaze.demo2.demo_main(config_args, input_path)
    
    # 调用方式2：指定输出路径
    sky_dehaze.demo2.demo_main(config_args, input_path, output_path)

    print(output_path+" done")


def sky_mask(input_path,output_path=''):
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(os.path.dirname(input_path), f"{name}_mask.jpg")

    sky_divide.sky_divide.detect(input_path, output_path)

    print(output_path+" done")

def merged(input_path,dcp_path='',sky_path='',mask_path=''):
    name, ext = os.path.splitext(filename)
    if(dcp_path == ''):
        dcp_path = os.path.join(os.path.dirname(input_path), f"{name}_DCP.jpg")
    if(sky_path == ''):
        sky_path = os.path.join(os.path.dirname(input_path), f"{name}_sky.jpg")
    if(mask_path == ''):
        mask_path = os.path.join(os.path.dirname(input_path), f"{name}_mask.jpg")
    result = merge.advanced_merged.advanced_merge(sky_path,dcp_path, mask_path, 
                        sigma=7, canny_sigma=1.5)
    # result = merge.advanced_merged_manual.advanced_merge(sky_path,dcp_path, mask_path, 
    #                     sigma=7, canny_sigma=1.5)
    output_path = os.path.join(os.path.dirname(input_path), f"{name}_merged.jpg")
    result.save(output_path)

    print(output_path+' done')

def evaluation(clear_path,compare_path,dcp_path='',sky_path='',mask_path='',merged_path=''):
    psnr_result = evaluate.psnr.compare_images_with_reference(clear_path,compare_path)
    ssim_result = evaluate.ssim.compare_images_with_reference(clear_path,compare_path)

    return [psnr_result,ssim_result]


def result_show(haze_path,clear_path,dcp_path='',sky_path='',mask_path='',merged_path=''):
    name, ext = os.path.splitext(filename)
    if(dcp_path == ''):
        dcp_path = os.path.join(os.path.dirname(haze_path), f"{name}_DCP.jpg")
    if(sky_path == ''):
        sky_path = os.path.join(os.path.dirname(haze_path), f"{name}_sky.jpg")
    if(mask_path == ''):
        mask_path = os.path.join(os.path.dirname(haze_path), f"{name}_mask.jpg")
    if(merged_path == ''):
        merged_path = os.path.join(os.path.dirname(haze_path), f"{name}_merged.jpg")
    
    output_path = os.path.join(os.path.dirname(haze_path), f"{name}_show.jpg")

    haze_img = Image.open(haze_path).convert('RGB')
    clear_img = Image.open(clear_path).convert('RGB')
    dcp_img = Image.open(dcp_path).convert('RGB')
    sky_img = Image.open(sky_path).convert('RGB')
    mask_img = Image.open(mask_path).convert('RGB')
    merged_img = Image.open(merged_path).convert('RGB')
    
    # 获取图片尺寸（假设所有图片尺寸相同）
    width, height = haze_img.size
    
    # 设置字体，尝试加载Arial，否则使用默认字体
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # 计算文本高度及边距
    test_str = 'hg'  # 测试字符串包含上下延伸的字符
    text_bbox = font.getbbox(test_str)
    text_height = text_bbox[3] - text_bbox[1] + 10  # 文本高度加10px边距
    
    # 计算布局尺寸
    row_height = height + text_height  # 每行总高度（图片+文字）
    total_width = 3 * width + 2 * 10   # 三张图片，间隔10px
    total_height = 2 * row_height + 50 # 两行，行间间隔50px
    
    # 创建新图像，背景白色
    new_img = Image.new('RGB', (total_width, total_height), color='white')
    draw = ImageDraw.Draw(new_img)
    
    # 图片列表及其对应名称
    images = [
        (haze_img, 'haze'),  (dcp_img, 'DCP'),(mask_img, 'mask'),
        (sky_img, 'sky'),  (merged_img, 'DCP+Wavelet U-Net'),(clear_img, 'clear'),
    ]
    
    # 遍历所有图片，排列并添加文字
    for index, (img, name) in enumerate(images):
        row = index // 3  # 确定行数（0或1）
        col = index % 3   # 确定列数（0、1、2）
        
        # 计算当前图片的粘贴位置
        x = col * (width + 10)
        y_start = row * (row_height + 50)  # 行间间隔50px
        
        # 粘贴图片到新图像
        new_img.paste(img, (x, y_start))
        
        # 计算文字位置（居中）
        text_x = x + width // 2
        text_y_center = y_start + height + text_height // 2
        draw.text((text_x, text_y_center), name, fill='black', font=font, anchor='mm')
    
    # 显示结果图像
    # new_img.show()

    new_img.save(output_path)

    


if __name__ == '__main__':
    input_path = r"C:\Users\lenovo\Desktop\bishe_final_pyqt\0151.jpg"
    clear_input_path = r'C:\Users\lenovo\Desktop\bishe_final_pyqt\0151.png'
    if len(sys.argv) < 2 :
        # print("未指定图片路径")
        pass
    else:
        input_path = sys.argv[1]

    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    # output_path = os.path.join(os.path.dirname(input_path), f"{name}_DCP.jpg")
    # print(filename,'\n',name,'\n',ext,'\n',output_path,end='\n')

    print(input_path)

    DCP(input_path)

    sky(input_path)

    # sky_mask(input_path)
    sky_mask(clear_input_path)
    merged(input_path)

    # 用法1, 对比俩张图片
    eva_result = evaluation(clear_input_path,os.path.join(os.path.dirname(input_path), f"{name}_merged.jpg"))
    # 用法2，对比一张图片和该图片下的所有图片
    # eva_result = evaluation(clear_input_path)

    # print(eva_result)

    result_show(input_path,clear_input_path)

    















