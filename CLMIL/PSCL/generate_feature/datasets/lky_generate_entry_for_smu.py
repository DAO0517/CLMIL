import os
import torch
import glob
from tqdm import tqdm

data_dir=r'E:\SMHE_FATURE\valid'
save_path=r'E:\SMHE_FATURE\feature\pt_files'
classname = os.listdir(data_dir)
for classname in classname:
    slide_ids = os.listdir(os.path.join(data_dir, classname))
    slide_path = []
    if not os.path.exists(os.path.join(save_path, classname)):
        os.makedirs(os.path.join(save_path, classname))
    for slide_id in slide_ids:
        slide_path.append(os.path.join(data_dir, classname, slide_id))
    for idx in range(len(slide_path)):
        slide_id = slide_path[idx]
        label = slide_path[idx].split('\\')[3]
        pt_path = glob.glob(os.path.join(slide_id, '*.pt'))  # 每一个slide下的patch pt
        feature_list = []
        for patch_feature_pt in pt_path:
            patch_feature = torch.load(patch_feature_pt)
            patch_feature = torch.from_numpy(patch_feature)
            feature_list.append(patch_feature)
        vectors = torch.cat(feature_list, dim=0)
        pt_path = os.path.join(save_path, classname, slide_path[idx].split('\\')[-1]+'.pt')
        torch.save(vectors, pt_path)