import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import time
from datasets.dataset_h5 import Dataset_All_Bags_lky, Patch_to_Bags_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
import argparse
from PSCL.utils.utils import collate_features_with_path, collate_features
from PSCL.utils.file_utils import save_hdf5
import h5py
from torchvision import transforms
from collections import OrderedDict
import torchvision.models as models

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def smu_transforms(pretrained=False):

	mean = (0.8648,0.7016,0.8058)
	std = (0.0861,0.1290,0.0892)

	trnsfrms_val = transforms.Compose(
					[
					 transforms.ToTensor(),
					 transforms.Normalize(mean = mean, std = std)
					]
				)

	return trnsfrms_val

def compute_w_loader(file_path, output_path, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
		custom_transforms: 归一化方法normal of slide（颜色归一化），None(pretrain为True，为imagenet的归一化，否则均为0.5，或自己定义)
	"""
	dataset = Patch_to_Bags_FP(bag_path=file_path, pretrained=pretrained,  custom_downsample=custom_downsample, target_patch_size=target_patch_size, custom_transforms=None)
	# x, y = dataset[0]
	kwargs = {'num_workers': 2, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features_with_path)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords, image_paths) in enumerate(loader):  # 返回的是（图和坐标） 这个loader的长度是一张wsi的子图
		with torch.no_grad():
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True, dtype=torch.float32)
			features = model(batch)
			features = features.cpu().numpy()
			asset_dict = {'features': features, 'coords': coords, 'image_paths': image_paths}
			save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
			mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_dir', type=str, default=r'G:\smu_meningoma_svs_cut')
parser.add_argument('--csv_path', type=str, default=r'E:\smu_meningoma_patch\pixibaoxing/pixibao_process_list_autogen.csv')
parser.add_argument('--feat_dir', type=str, default=r'G:\512_weeksup_round2_smu_meningoma_patch_feature')  # 保存的路径
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1, help='The value is -1, mean the origin size')
parser.add_argument('--checkpoint_path', type=str, default=r'E:\2021022219\lkycode\model_checkpoint\week_round2_model_best_smu_2.pth', help='the path of checkpoint')
args = parser.parse_args()


if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags_lky(args.data_dir)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	print('loading model checkpoint')
	if args.checkpoint_path is not None:
		model = resnet50_baseline(pretrained=False)
		state_dict = OrderedDict()
		checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
		checkpoint_state_dict = checkpoint['state_dict']
		for k, v in checkpoint_state_dict.items():
			if (k.startswith('encoder_q') and not k.startswith('encoder_q.fc') and
					not k.startswith('encoder_q.layer4')):
				state_dict[k[len('encoder_q.'):]] = v
		model.load_state_dict(state_dict, strict=False)
	else:
		model = resnet50_baseline(pretrained=True)
	model = model.to(device)
	
	# print_network(model)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)

	model.eval()

	for class_name in os.listdir(args.data_dir):
		class_path = os.path.join(args.data_dir, class_name)
		for bag_name in os.listdir(class_path):
			bag_path = os.path.join(class_path, bag_name)
			if not args.no_auto_skip and bag_name+'.pt' in dest_files:
				print('skipped {}'.format(bag_name))
				continue

			output_path = os.path.join(args.feat_dir, 'h5_files', bag_name+'.h5')
			time_start = time.time()

			output_file_path = compute_w_loader(bag_path, output_path,
			model=model, batch_size=args.batch_size, verbose=1, print_every=20,
			pretrained=True, custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
			time_elapsed = time.time() - time_start
			print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
			file = h5py.File(output_file_path, "r")

			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)
			features = torch.from_numpy(features)
			bag_base, _ = os.path.splitext(bag_name+'.pt')
			torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))




