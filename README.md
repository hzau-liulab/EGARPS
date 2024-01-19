# EGARPS
# Description
NABind is a novel structure-based method to predict DNA/RNA-binding residues by leveraging deep learning and template approaches.  
![image](img/flow.jpg)  

# Third-party software needed
GHECOM https://pdbj.org/ghecom/  
DSSP https://swift.cmbi.umcn.nl/gv/dssp/DSSP_5.html  
NACCESS http://www.bioinf.manchester.ac.uk/naccess/  
DSSR https://x3dna.org/  

# Usage
## 1. Download pre-trained model
Download pre-trained model from this link and place the pth file into the ***weights*** folder.  
## 2. Configuration
    conda env create -f environment.yml  
    conda activate EGARPS  
Edit the config file (***scripts/config.json***) to suit your needs  
## 3. Run EGARPS for prediction
    cd ./scripts/
    python predict.py ../example/complex1.pdb

# Citation
Accurately evaluate RNA-protein complex structures through multi-view features and neural networks. *Submitted*, 2024.
