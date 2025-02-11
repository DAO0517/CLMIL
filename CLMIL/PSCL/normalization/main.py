import numpy as np
import cv2
import matplotlib.pyplot as plt
from macenko import MacenkoNormalizer
from vahadane import VahadaneNormalizer

def standard_transfrom(standard_img,method = 'M'):
    if method == 'V':
        stain_method = VahadaneNormalizer()
        stain_method.fit(standard_img)
    else:
        stain_method = MacenkoNormalizer()
        stain_method.fit(standard_img)
    return stain_method

def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # opencv default color space is BGR, change it to RGB
    p = np.percentile(img, 90)
    img = np.clip(img * 255.0 / p, 0, 255).astype(np.uint8)
    return img

if __name__ == '__main__':
    path='/media/totem_disk/totem/wangshuhuan/tmp/patch/LumA/TCGA-3C-AALK_123.png'
    sttd=read_image(path)
    # plt.imshow(sttd)
    # plt.show()
    stain_method = standard_transfrom(sttd, method='M')
    img=cv2.imread('/media/totem_disk/totem/wangshuhuan/tmp/ceshi/TCGA-5L-AAT0_63.png')
    img2 = stain_method.transform(img)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.show()