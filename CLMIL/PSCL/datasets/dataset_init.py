import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2
from collections import defaultdict
import torchvision.transforms as transforms
import pickle
import glob

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

class InsDataset(Dataset):
    def __init__(self, root_dir, bag_names_file, transform, pseudo_label=None, threshold=0.7, witness_rate=None,
                 labelroot=None):
        super().__init__(root_dir, bag_names_file, transform, witness_rate, labelroot)
        self.tiles = sum([[k + v for v in vs] for k, vs in self.bag2tiles.items()], [])

        if 'train' in bag_names_file or 'val' in bag_names_file:
            gt_path = labelroot + '/annotation/gt_ins_labels_train.p'
            self.train_stat = True
        else:
            gt_path = labelroot + '/annotation/gt_ins_labels_test.p'
            self.train_stat = False
        self.gt_label = pickle.load(open(gt_path, 'rb'))

        if pseudo_label:
            self.pseudo_label = pickle.load(open(pseudo_label, 'rb'))
        else:
            self.pseudo_label = None
        self.threshold = threshold

    def update_pseudo_label(self, label_path):
        self.pseudo_label = pickle.load(open(label_path, 'rb'))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        # print(self.root_dir)
        # print(self.tiles[index])
        # tile_dir = os.path.join(self.root_dir, self.tiles[index])
        tile_dir = self.root_dir + self.tiles[index]
        # print(tile_dir)
        img = self._get_data(tile_dir)
        img = transforms.functional.to_tensor(img)
        # two augmentations

        img1 = self.transform(img)
        if self.train_stat:
            img2 = self.transform(img)
        # return img, index

        temp_path = tile_dir
        # print(temp_path)

        bag_label = 1 if "tumor" in temp_path else 0

        label = bag_label

        #        /single/training/normal_001/24_181.jpeg
        file_list = temp_path.split('.')[0]
        file_list = file_list.split('/')[-2:]

        if 'train' in tile_dir:
            slide_name = "/training/" + file_list[0]  # normal_001
        else:
            slide_name = "/testing/" + file_list[0]  # test_001
        patch_name = file_list[1]  # 24_181

        if self.pseudo_label:

            pseudo_label = self.pseudo_label[slide_name][patch_name]
            # print(pseudo_label)
            if pseudo_label > self.threshold and label == 1:
                label = 1
            else:
                label = 0

        if self.train_stat:
            return img1, img2, torch.LongTensor(np.array([bag_label])), torch.LongTensor(
                np.array([label])), torch.LongTensor([self.gt_label[slide_name][patch_name]]), slide_name, patch_name
        else:
            return img1, torch.LongTensor(np.array([bag_label])), torch.LongTensor(np.array([label])), torch.LongTensor(
                [self.gt_label[slide_name][patch_name]]), slide_name, patch_name

class Patch_to_Bags_FP(Dataset):

	def __init__(self, bag_path=None, pretrained=False, custom_transforms=None, custom_downsample=1, target_patch_size=-1):
		self.bag_path = bag_path
		image_paths = glob.glob(os.path.join(bag_path, '*.png'))
		self.image_paths = image_paths
		self.custom_transforms = custom_transforms
		if not self.custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
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
		encoded_coord = np.array([c.encode('utf-8') for c in coord])
		img = Image.open(image_path).convert('RGB')
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		else:
			img = self.roi_transforms(img).unsqueeze(0)  # 将原始图像张量由形状 (C, H, W)（通道数、高度、宽度）变成了 (1, C, H, W)
		return img, encoded_coord  # 从 DataFrame 中的 'slide_id' 列中选择特定行 idx 处的值