# PLMAM-PLA: a method using pretrained language models and attention mechanisms for protein-ligand binding affinity prediction
## Table of Contents

1. [Introduction](#introduction)
2. [Python Environment](#python-environment)
3. [Project Structure](#Project-Structure)
   1. [Dataset](#Dataset)
   2. [Model](#Model)
   3. [script](#script)
---
## 1. Introduction

We
developed a novel sequence-based deep learning method, called PLMAM-PLA, to predict protein-ligand binding affinity.
PLMAM-PLA is constructed by integrating local and global sequence features. The PLMAM-PLA model consists of a
feature extraction module,a feature enhancement module, a feature enhancement module, and an output module. The
feature extraction module uses a CNN sub-module to obtain the initial local features of the sequence and a pretrained
language model to obtain the global features. The feature enhancement module extracts higher-order local and global
features. The feature fusion module learns protein-ligand interactions and integrates all the features. The proposed
model is trained and tested using the PDBbind v2016 dataset. We compared PLMAM-PLA with the latest state-of-theart methods and analyzed the effectiveness of different parts of the model. The results show that the proposed model
outperforms other deep learning models.


## 2. Python Environment

Python 3.9 and packages version:

- pytorch==2.2.1
- tqdm==4.66.2                            
- torchvision==0.17.1    
- transformers==4.22.2
- numpy==1.26.4
- pandas==2.1.4
- scikit-learn==1.4.1
- scipy==1.12.0 

## 3. Project Structure

### 3.1 **Dataset**

   The PDBbind database contains a set of experimentally validated protein-ligand binding complexes from the Protein Data Bank, where the protein-ligand binding affinities are expressed as -logKi, -logKd, or -logIC50. In this paper, we collect data from PDBbind version 2016 , which consists of the general set, the refined set, and the core 2016 set. To avoid overlap between these datasets, 290 protein-ligand complexes from the core 2016 set were removed from the refined set. Then, the resulting general set contains 9221 complexes, the refined set contains 3685 complexes, and the core 2016 set contains 290 complexes. Then, 1000 complexes in the refined set are randomly selected as the validation set, the remaining complexes are combined with all the complexes in the general set to form the training set, and the core 2016 set is used to be the testing set. As a result, there are 11906 training samples, 1000 validation samples and 290 test samples, where each sample comprises a protein sequence, a ligand SMILES and a protein-ligand binding affinity value.

### 3.2 **Model**
   -  The overall architectures of PLMAM-PLA is presented in the following figure, which consists of a feature extraction module, a feature enhancement module, a feature fusion module and an output module.
   -  ![Model Architecture](https://github.com/SAJ-2001/PLMAM-PLA/blob/main/PLMAMPLA.jpg)
   -  trained_model.pt is the PLMAM-PLA model that is trained on the training subest of the PDBbind dataset.
   -  The ESM-2 model is available at (https://github.com/facebookresearch/esm) and Molformer model is available at (https://huggingface.co/ibm/MoLFormer-XL-both-10pct).
   -   To load the model from Huggingface, we can use the following code:
```python
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

smiles = ["Cn1c(=O)c2c(ncn2C)n(C)c1=O", "CC(=O)Oc1ccccc1C(=O)O"]
inputs = tokenizer(smiles, padding=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
outputs.pooler_output
```

### 3.3 **data**
   -   affinity_data.csv is the affinity value.
   -   test_seq.csv, training_seq.csv and validation_seq.csv are protein sequences.
   -   test_smi.csv, training_smi.csv and validation_smi.csv are ligand SMILES.
   - `get_esm.py` converts protein sequences into token embeddings.
   - `get_sm.py` converts ligand SMILES into token embeddings.
### 3.4 **script**
   -   To train the model, we can run `main.py` script using the train and valid dataset.
   -   We can also run `test.py` to test the model.
   - `dataset.py` is the data preparation phase.
   - `cross_attention.py` implements cross attention mechanisms.
   - `model.py` implements the PLMAM-PLA which consists of a feature extraction module, a feature enhancement module, a feature fusion module and an output module.
