import os
from PIL import Image
import numpy as np
import cv2

class HazeRemoval:
    def __init__(self, filename, omega=0.85, r=40, d_bilateral=9, sigma_color=75, sigma_space=75):
        self.filename = filename
        self.omega = omega
        self.r = r
        self.d_bilateral = d_bilateral          # 双边滤波邻域直径
        self.sigma_color = sigma_color          # 颜色空间标准差
        self.sigma_space = sigma_space          # 坐标空间标准差
        self.eps = 10 ** (-3)
        self.t = 0.1

    def _ind2sub(self, array_shape, ind):
        rows = (ind.astype('int') // array_shape[1])
        cols = (ind.astype('int') % array_shape[1])
        return (rows, cols)

    def _rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def haze_removal(self):
        oriImage = np.array(Image.open(self.filename).convert('RGB'))
        img = oriImage.astype(np.double) / 255.0
        grayImage = self._rgb2gray(img)

        darkImage = img.min(axis=2)
        i, j = self._ind2sub(darkImage.shape, darkImage.argmax())
        A = img[i, j, :].mean()
        transmission = 1 - self.omega * darkImage / A

        grayImage = grayImage.astype(np.float32)
        transmission = transmission.astype(np.float32)

        # 导向滤波
        transmissionFilter = cv2.ximgproc.guidedFilter(grayImage, transmission, self.r, self.eps)
        
        # 双边滤波
        transmissionFilter = cv2.bilateralFilter(
            transmissionFilter,
            d=self.d_bilateral,
            sigmaColor=self.sigma_color,
            sigmaSpace=self.sigma_space
        )
        
        transmissionFilter[transmissionFilter < self.t] = self.t

        resultImage = np.zeros_like(img)
        for i in range(3):
            resultImage[:, :, i] = (img[:, :, i] - A) / transmissionFilter + A

        resultImage = np.clip(resultImage, 0, 1)
        return Image.fromarray((resultImage * 255).astype(np.uint8))

if __name__ == '__main__':
    input_path = r"0151.jpg"
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(os.path.dirname(input_path), f"{name}_DCP.jpg")
        
    hz = HazeRemoval(input_path)
    result = hz.haze_removal()
    result.save(output_path)
    result.show()