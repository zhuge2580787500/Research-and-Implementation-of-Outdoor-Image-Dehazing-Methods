import os
import sys
import DCP_dehaze.DCP_filter
from sky_dehaze.config import get_config
import sky_dehaze.demo2
import sky_divide.sky_divide
import merge.advanced_merged
import evaluate.psnr
import evaluate.ssim


def DCP(input_path, output_path=''):
    name, ext = os.path.splitext(os.path.basename(input_path))
    output_path = os.path.join(os.path.dirname(input_path), f"{name}_DCP.jpg")
    result = DCP_dehaze.DCP_filter.HazeRemoval(input_path).haze_removal()
    result.save(output_path)
    print(output_path + " done")
    return output_path


def sky(input_path, output_path=''):
    config_args, _ = get_config()
    config_args.base_size = 480  # 可调整的基础尺寸
    config_args.use_gpu = True
    config_args.gpu = 0
    config_args.model_dir = r'C:\Users\lenovo\Desktop\bishe_final\sky_dehaze\model'
    config_args.ckpt = 'dehaze_chromatic_100.pkl'

    name, ext = os.path.splitext(os.path.basename(input_path))
    output_path = os.path.join(os.path.dirname(input_path), f"{name}_sky.jpg")

    # 调用方式2：指定输出路径
    sky_dehaze.demo2.demo_main(config_args, input_path, output_path)

    print(output_path + " done")
    return output_path


def sky_mask(input_path, output_path=''):
    name, ext = os.path.splitext(os.path.basename(input_path))
    output_path = os.path.join(os.path.dirname(input_path), f"{name}_mask.jpg")

    sky_divide.sky_divide.detect(input_path, output_path)

    print(output_path + " done")
    return output_path


def merged(input_path, dcp_path='', sky_path='', mask_path=''):
    name, ext = os.path.splitext(os.path.basename(input_path))
    if dcp_path == '':
        dcp_path = os.path.join(os.path.dirname(input_path), f"{name}_DCP.jpg")
    if sky_path == '':
        sky_path = os.path.join(os.path.dirname(input_path), f"{name}_sky.jpg")
    if mask_path == '':
        mask_path = os.path.join(os.path.dirname(input_path), f"{name}_mask.jpg")
    result = merge.advanced_merged.advanced_merge(sky_path, dcp_path, mask_path,
                                                sigma=7, canny_sigma=1.5)
    output_path = os.path.join(os.path.dirname(input_path), f"{name}_merged.jpg")
    result.save(output_path)

    print(output_path + ' done')
    return output_path


def evaluation(input_path, dcp_path='', sky_path='', mask_path='', merged_path=''):
    evaluate.psnr.compare_images_with_reference(input_path)
    evaluate.ssim.compare_images_with_reference(input_path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        input_folder = r"C:\Users\lenovo\Desktop\bishe_final_edit"
        clear_folder = r'C:\Users\lenovo\Desktop\bishe_final_edit'
    else:
        input_folder = sys.argv[1]
        clear_folder = sys.argv[1]

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            clear_input_path = os.path.join(clear_folder, os.path.splitext(filename)[0] + '.png')

            dcp_output_path = DCP(input_path)
            sky_output_path = sky(input_path)
            mask_output_path = sky_mask(input_path)
            merged_output_path = merged(input_path, dcp_output_path, sky_output_path, mask_output_path)
            evaluation(clear_input_path)

            # 删除 sky、mask 和 dcp 图片
            try:
                os.remove(dcp_output_path)
                os.remove(sky_output_path)
                os.remove(mask_output_path)
                print(f"已删除 {dcp_output_path}, {sky_output_path}, {mask_output_path}")
            except FileNotFoundError:
                print("文件未找到，无法删除。")
            except Exception as e:
                print(f"删除文件时出现错误: {e}")