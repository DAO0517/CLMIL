# Optimized Multiple Instance Learning for Brain Tumor Classification Using Weakly Supervised Contrastive Learning
This repository contains the implementation of CLMIL, a two-stage framework for brain tumor classification. The framework consists of:

1. CDMIL (Cross-Detection MIL Aggregator): A model for brain tumor classification using Multiple Instance Learning (MIL).

2. PSCL (Contrastive Learning Model based on Pseudo-Labels): A contrastive learning model to optimize the feature encoder using pseudo-labels.
The model is trained on a single RTX 3080 GPU.
## Requirements
To run the code, you need the following dependencies:
- Python 3.7+
- PyTorch 1.8+
- torchvision
- torch_geometric
- numpy
- scikit-learn
- spams

(Add any other dependencies your project requires)
 
## Model training and testing
Our framework CLMIL consists of two stages: a cross-detection MIL aggregator (CDMIL) for brain tumor classification and a contrastive learning model based on pseudo-labels (PSCL) for optimizing feature encoder. Generally, we train the model with one RTX 3080 GPU. 
<div align="center">
  <img src="figures/fig1.png">
</div>

## Dataset Preparation
The datasets used in this study are accessible through The Cancer Genome Atlas (TCGA) at [https://portal.gdc.cancer.gov/]  and the CAMELYON16 challenge at [https://camelyon16.grand-challenge.org/]. The dataset of Meningiomas is available from the corresponding authors upon reasonable request.
  ### Data Preprocessing
All Whole Slide Images (WSIs) are processed into small patches of size 512x512 pixels at 20X magnification. These patches are used for feature extraction and model training.
### Step 1: Extract Patch Features
To extract features from the patches, navigate to the generate_feature directory under PSCL and run the following command:
```
cd ../CLMIL/PSCL/generate_feature
python extract_features_moco.py
```
The patch feature extractor is a truncated ResNet50 model, which is self-supervised pre-trained on the CAMELYON16 dataset. The features files of WSIs in our work is available at [https://drive.google.com/drive/folders/1wCvUZBantttrAcnXjC5GVSCN53szn1-u?usp=drive_link].
### Step 2: Train CDMIL (First Stage)
Navigate to the CDMIL directory and run the following command to train the CDMIL model:
```
   cd ../CLMIL/CDMIL
   python train.py
```
### Step 3: Extract Anchor Samples and Pseudo-Labels from IPAM
Once the CDMIL model is trained, you can extract anchor samples and pseudo-labels from the IPAM. These will be used as training data for the PSCL stage. Run the following script to extract the data and save the extracted anchor samples and pseudo-labels:
```
   cd ../CLMIL/PSCL
   python select_the_sample.py
```
### Step 4: Train PSCL (Second Stage)
Navigate to the PSCL directory and run the following command to train the PSCL model using the extracted anchor samples and pseudo-labels:
```
   cd ../CLMIL/PSCL
   python finetune_extractor.py
```
After training, the PSCL model will save the optimized feature encoder in the specified model_path.
### Step 5: Extract Optimized Features
Once the feature encoder is optimized, you can load the parameters of optimized feature encoder and extract features from the dataset using the following command:
```
cd ../CLMIL/PSCL/generate_feature
python extract_features_moco.py
```
### Step 6: Retrain CDMIL with Optimized Features
Finally, navigate back to the CDMIL directory and retrain the CDMIL model using the optimized features:
```
   cd ../CLMIL/CDMIL
   python train.py
```
The features files of WSIs in our work is available at [https://drive.google.com/drive/folders/1wCvUZBantttrAcnXjC5GVSCN53szn1-u?usp=drive_link].

The parameters of models in our work is available at [https://drive.google.com/drive/folders/1Nh3vy8-hM82-yRnNZXbrWpU9Ewu_67Zc?usp=drive_link].
## Contact
If you have any question, please feel free to contact us. 
