# EGARPS
# Description
EGARPS is a novel tool for evaluating protein-RNA complex structures, which achieves high accuracy by integrating well-designed intermolecular and intramolecular modules with a series of biophysically inspired descriptors. Extensive evaluations illustrated that EGARPS significantly outperformed existing methods on both bound and unbound datasets. Our top 1 success rate was more than twice that of the CNN-based approach. Besides, EGARPS can improve the performance of de novo RNA-protein complex prediction.  
![image](img/F1.png)  

# Datasets
EGARPS was evaluated on a bound dataset as well as two widely used protein-RNA docking benchmark datasets. The source data can be downloaded from the following links:  
https://github.com/Zhaolab-GitHub/DRPScore_v1.0/blob/main/DRPScore_4DCNN.zip  
https://zoulab.dalton.missouri.edu/RNAbenchmark/index.htm  
https://life.bsc.es/pid/protein-rna-benchmark/  

# Third-party software needed
Some third-party software has been integrated into the EGARPS pipeline, primarily for structure analysis. DSSP and DSSR are used for identifying the secondary structures of proteins and RNA, respectively, while NACCESS and GHECOM are employed for tertiary structure analysis of proteins and RNA. The download links are provided as follows:  
GHECOM https://pdbj.org/ghecom/  
DSSP https://swift.cmbi.umcn.nl/gv/dssp/DSSP_5.html  
NACCESS http://www.bioinf.manchester.ac.uk/naccess/  
DSSR https://x3dna.org/  
Users need to download and compile the software on their own machines.

# Usage
## 1. Download pre-trained model
Download pre-trained model from [this link](https://drive.google.com/file/d/1Fux72Ayp1g_k7yxytfA0ki_WSoxX51Aa/view?usp=drive_link) and place the pth file into the ***weights*** folder.  
## 2. Configuration
    conda env create -f environment.yml  
    conda activate EGARPS  
&  
Edit the config file (***scripts/config.json***) to suit your needs  
## 3. Run EGARPS for prediction
    cd ./scripts/
    python predict.py ../example/complex1.pdb

# Citation
Graph learning-based scoring of RNA-protein complex structures using EGARPS. *Submitted*, 2024.
