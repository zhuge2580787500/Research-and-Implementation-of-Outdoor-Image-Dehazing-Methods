import os
import sys
import DCP_dehaze.DCP_filter
from sky_dehaze.config import get_config
import sky_dehaze.demo2
import sky_divide.sky_divide
import merge.advanced_merged
import evaluate.psnr
import evaluate.ssim


def DCP(input_path,output_path=''):
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(os.path.dirname(input_path), f"{name}_DCP.jpg")
    result = DCP_dehaze.DCP_filter.HazeRemoval(input_path).haze_removal()
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
    output_path = os.path.join(os.path.dirname(input_path), f"{name}_merged.jpg")
    result.save(output_path)

    print(output_path+' done')

def evaluation(input_path,dcp_path='',sky_path='',mask_path='',merged_path=''):
    evaluate.psnr.compare_images_with_reference(input_path)
    evaluate.ssim.compare_images_with_reference(input_path)

if __name__ == '__main__':
    input_path = r"C:\Users\lenovo\Desktop\bishe_final\0151.jpg"
    clear_input_path = r'C:\Users\lenovo\Desktop\bishe_final\0151.png'
    if len(sys.argv) < 2 :
        # print("未指定图片路径")
        pass
    else:
        input_path = sys.argv[1]

    filename = os.path.basename(input_path)
    # name, ext = os.path.splitext(filename)
    # output_path = os.path.join(os.path.dirname(input_path), f"{name}_DCP.jpg")
    # print(filename,'\n',name,'\n',ext,'\n',output_path,end='\n')

    print(input_path)

    DCP(input_path)

    sky(input_path)

    sky_mask(input_path)

    merged(input_path)
    
    evaluation(clear_input_path)













