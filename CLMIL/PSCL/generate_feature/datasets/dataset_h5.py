from __future__ import print_function, division
import os
import glob
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
import cv2

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import h5py

from random import randrange
from PSCL.normalization.macenko import MacenkoNormalizer
from PSCL.normalization.vahadane import VahadaneNormalizer

def standard_transfrom(standard_img,method = 'M'):
    if method == 'V':
        stain_method = VahadaneNormalizer()
        stain_method.fit(standard_img)
    else:
        stain_method = MacenkoNormalizer()
        stain_method.fit(standard_img)
    return stain_method

def eval_transforms(pretrained=False):
	if pretrained:
		mean = (0.485, 0.456, 0.406)
		std = (0.229, 0.224, 0.225)

	else:
		mean = (0.5,0.5,0.5)
		std = (0.5,0.5,0.5)

	trnsfrms_val = transforms.Compose(
					[
					 transforms.ToTensor(),
					 transforms.Normalize(mean = mean, std = std)
					]
				)

	return trnsfrms_val

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		pretrained=False,
		custom_transforms=None,
		target_patch_size=-1,
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.pretrained=pretrained
		if target_patch_size > 0:
			self.target_patch_size = (target_patch_size, target_patch_size)
		else:
			self.target_patch_size = None

		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['imgs']
		for name, value in dset.attrs.items():
			print(name, value)

		print('pretrained:', self.pretrained)
		print('transformations:', self.roi_transforms)
		if self.target_patch_size is not None:
			print('target_size: ', self.target_patch_size)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		pretrained=False,
		custom_transforms=None,
		custom_downsample=1,
		target_patch_size=-1
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
		"""
		self.pretrained=pretrained
		self.wsi = wsi
		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']  # 使用 .attrs，你可以访问与数据集关联的属性
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			if target_patch_size > 0:
				self.target_patch_size = (target_patch_size, ) * 2
			elif custom_downsample > 1:
				self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
			else:
				self.target_patch_size = None
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('target patch size: ', self.target_patch_size)
		print('pretrained: ', self.pretrained)
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]  # 坐标
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)  # 将原始图像张量由形状 (C, H, W)（通道数、高度、宽度）变成了 (1, C, H, W)
		return img, coord

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]  # 从 DataFrame 中的 'slide_id' 列中选择特定行 idx 处的值


class Dataset_All_Bags_lky(Dataset):

	def __init__(self, data_path):
		self.name_list = os.listdir(data_path)

	def __len__(self):
		return len(self.name_list)

	def __getitem__(self, idx):
		return self.name_list[idx]  # 从 DataFrame 中的 'slide_id' 列中选择特定行 idx 处的值


class Patch_to_Bags_FP(Dataset):

	def __init__(self, bag_path=None, pretrained=False, custom_transforms=None, custom_downsample=1, target_patch_size=-1):
		self.bag_path = bag_path
		image_paths = glob.glob(os.path.join(bag_path, '*.png'))
		self.image_paths = image_paths
		self.custom_transforms = custom_transforms
		if not self.custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		elif self.custom_transforms=='normal of slide':
			sttd = read_image(r'G:\smu_meningoma_svs_cut\hunhexing\1709417-he1 2_20\16_52.png')
			method = standard_transfrom(sttd, method='M')
			self.roi_transforms = method
		else:
			self.roi_transforms = self.custom_transforms
		if target_patch_size > 0:
			self.target_patch_size = (target_patch_size, target_patch_size)
		else:
			self.target_patch_size = None

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		image_path = self.image_paths[idx]
		image_name = image_path.split('.png')[0].split('\\')[-1]
		y_coord = image_name.split('_')[0]
		x_coord = image_name.split('_')[-1]
		coord = np.array([y_coord, x_coord])
		encoded_coord = np.array([c.encode('utf-8') for c in coord])  # 编码为了save_h5py
		encoded_path = np.array(image_path.encode('utf-8'))
		img = Image.open(image_path).convert('RGB')
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		else:
			img = self.roi_transforms(img).unsqueeze(0)  # 将原始图像张量由形状 (C, H, W)（通道数、高度、宽度）变成了 (1, C, H, W)
		return (img, encoded_coord, encoded_path)



def read_image(path):
	img=Image.open(path).convert('RGB')
	img = np.array(img, dtype=np.uint8)
	p = np.percentile(img, 90)
	img = np.clip(img * 255.0 / p, 0, 255).astype(np.uint8)
	return img

