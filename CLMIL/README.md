# Optimized Multiple Instance Learning for Brain Tumor Classification Using Weakly Supervised Contrastive Learning
This repository contains the implementation of CLMIL, a two-stage framework for brain tumor classification. The framework consists of:

1. CDMIL (Cross-Detection MIL Aggregator): A model for brain tumor classification using Multiple Instance Learning (MIL).

2. PSCL (Contrastive Learning Model based on Pseudo-Labels): A contrastive learning model to optimize the feature encoder using pseudo-labels.
The model is trained on a single RTX 3080 GPU.
## Requirements
 ### Installation
Please install pytorch version >=1.2
 ### Dataset Preparation
The datasets used in this study are accessible through The Cancer Genome Atlas (TCGA) at [https://portal.gdc.cancer.gov/] and the CAMELYON16 challenge at [https://camelyon16.grand-challenge.org/]. The dataset of Meningiomas is available from the corresponding authors upon reasonable request.
 ## Model training and testing
Our framework CLMIL consists of two stages: a cross-detection MIL aggregator (CDMIL) for brain tumor classification and a contrastive learning model based on pseudo-labels (PSCL) for optimizing feature encoder. Generally, we train the model with one RTX 3080 GPU. 
<div align="center">
  <img src="figures/fig1.png">
</div>

### Training the CDMIL
```
   cd ../CLMIL/CDMIL
   python train.py
```
### Training the PSCL

## Contact
If you have any question, please feel free to contact us. 
