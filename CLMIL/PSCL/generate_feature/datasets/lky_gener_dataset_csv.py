import pandas as pd
import os
import csv

combined_data = pd.DataFrame()
folder_path = r"E:\SMHE_FATURE\feature\pt_files"
pixibaoxing_name = os.listdir(r'E:\SMHE_FATURE\feature\pt_files_subtype\0')
xianweixing_name = os.listdir(r'E:\SMHE_FATURE\feature\pt_files_subtype\1')
hunhexing_name = os.listdir(r'E:\SMHE_FATURE\feature\pt_files_subtype\2')
for i, filename_h5 in enumerate(os.listdir(folder_path)):
    filename = filename_h5.split('.')[0]
    if filename_h5 in pixibaoxing_name:
        label = 'pixibaoxing'
    if filename_h5 in xianweixing_name:
        label = 'xianweixing'
    if filename_h5 in hunhexing_name:
        label = 'hunhexing'

    data = {'case_id': i, 'slide_id': filename, 'label': label}
    combined_data = combined_data.append(data, ignore_index=True)
'''
# 生成camelyon16test数据的对应标签
test_label_dict = {}
with open(r'F:\CAMELYON16\testing\reference.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    # 遍历每一行数据
    for row in csv_reader:
        if len(row) >= 2:
            key = row[0]  # 第一列作为键
            value = row[1]  # 第二列作为值
            test_label_dict[key] = value

# 遍历每个文件名
for i, filename in enumerate(os.listdir(folder_path)):
    filename = filename.split('.')[0]
    if 'normal' in filename:
        label = 'normal_tissue'
    if 'tumor' in filename:
        label = 'tumor_tissue'
    if 'test' in filename:
        label = test_label_dict[filename]
        if label == 'Normal':
            label = 'normal_tissue'
        else:
            label = 'tumor_tissue'

    data = {'case_id': i, 'slide_id': filename, 'label': label}
    combined_data = combined_data.append(data, ignore_index=True)
'''
# 指定保存整合后数据的文件路径
output_file = r'E:\2021022219\CLAM-master\dataset_csv/meningioma_tumor_subtyping_dummy_clean_0.1withmoco.csv'

# 将整合后的数据保存到CSV文件中
combined_data.to_csv(output_file, index=False)