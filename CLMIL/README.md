# Optimized Multiple Instance Learning for Brain Tumor Classification Using Weakly Supervised Contrastive Learning
## Requirements
 ### Installation
Please install pytorch version >=1.2
 ### Dataset Preparation
The datasets used in this study are accessible through The Cancer Genome Atlas (TCGA) at [https://portal.gdc.cancer.gov/] and the CAMELYON16 challenge at [https://camelyon16.grand-challenge.org/]. The dataset of Meningiomas is available from the corresponding authors upon reasonable request.
 ## Model training and testing
Our framework CLMIL consists of two stages: a cross-detection MIL aggregator (CDMIL) for brain tumor classification and a contrastive learning model based on pseudo-labels (PSCL) for optimizing feature encoder. Generally, we train the model with one RTX 3080 GPU. 
 ~~~~~~~~~~~~~~~~~~
   python train.py 
 ~~~~~~~~~~~~~~~~~~

## Contact
If you have any question, please feel free to contact us. 
