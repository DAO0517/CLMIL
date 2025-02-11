import os

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def augment_hsv(img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    """
    hsv增强  处理图像hsv，不对label进行任何处理
    :param img: 待处理图片  BGR [736, 736]
    :param h_gain: h通道色域参数 用于生成新的h通道
    :param s_gain: h通道色域参数 用于生成新的s通道
    :param v_gain: h通道色域参数 用于生成新的v通道
    :return: 返回hsv增强后的图片 img
    """

    # 从-1~1之间随机生成3随机数与三个变量进行相乘
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    # 分别针对hue, sat以及val生成对应的Look-Up Table（LUT）查找表
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    # 使用cv2.LUT方法利用刚刚针对hue, sat以及val生成的Look-Up Table进行变换
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    aug_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # 这里源码是没有进行return的，不过我还是觉得return一下比较直观了解
    return aug_img

if __name__ == '__main__':
    #获取图像列表，此处为原始图像所在路径
    files = os.listdir("/media/totem_disk/totem/wangshuhuan/tmp/ceshi")
    s=0
    m=0
    for file in files:
        #按照获取的列表依次读取列表，路径同上
        img = io.imread('/media/totem_disk/totem/wangshuhuan/tmp/ceshi/' + os.path.splitext(file)[0] + '.png')
        img_out = augment_hsv(img)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(img_out)
        plt.show()
        #s = s + 1
        #路径为结果保存路径
        imageio.imwrite("/media/totem_new/totem/Breast_Pathology/patch_mask_512/TCGA_cls/Basal/" + os.path.splitext(file)[0][:4] + "_{}.png".format(s), img_out)

    #print('已处理完{}张照片'.format(s))
