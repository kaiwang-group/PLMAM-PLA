from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import torch.utils.data

CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}

CHARPROTSET = {"A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6,
               "H": 7, "I": 8, "K": 9, "L": 10, "M": 11,
               "N": 12, "P": 13, "Q": 14, "R": 15, "S": 16,
               "T": 17, "V": 18, "W": 19,
               "Y": 20, "X": 21
               }
def label_sequence(line, MAX_SEQ_LEN):
    X = np.zeros(MAX_SEQ_LEN, dtype=int)
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = CHARPROTSET[ch]
    return X


def label_smiles(line, max_smi_len):
    X = np.zeros(max_smi_len, dtype=int)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch]
    return X

class MyDataset(Dataset):
    def __init__(self, data_path, phase,max_seq_len,max_smi_len):
        super(MyDataset, self).__init__()
        data_path = Path(data_path)
        prots_embedding = []
        smiless_embedding = []

        affinity_df = pd.read_csv(data_path / 'affinity_data.csv', sep='\t')
        ligands_df = pd.read_csv(data_path / f"{phase}_smi.csv")
        prots_df = pd.read_csv(data_path / f"{phase}_seq.csv")


        # Check if pdbid exists in all dataframes
        common_pdbids = set(affinity_df['pdbid']) & set(ligands_df['pdbid']) & set(prots_df['pdbid'])
        # Filter ligands and prots based on common_pdbids
        ligands = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows() if i["pdbid"] in common_pdbids}
        prots = {i["pdbid"]: i["seq"] for _, i in prots_df.iterrows() if i["pdbid"] in common_pdbids}
      
        affinity = {}
        self.affinity = affinity
        for pdbid in common_pdbids:
            affinity_val = affinity_df.loc[affinity_df['pdbid'] == pdbid, 'affinity'].values
            if len(affinity_val) > 0:
                self.affinity[pdbid] = affinity_val[0]

        smiles_feature = {}
        for pdbid in  ligands:
            npy_file_path = data_path/f'Molformer1/{pdbid}.npy'
            smiles_feature[pdbid] = np.load(npy_file_path, allow_pickle=True)
            smiles_embedding = smiles_feature[pdbid]
            smiless_embedding.append(smiles_embedding)

        pro_feature = {}
        for pdbid in  prots:
            npy_file_path = data_path/f'token embedding1-1000-8M/{pdbid}.npy'
            pro_feature[pdbid] = np.load(npy_file_path, allow_pickle=True)
            pro_embedding = pro_feature[pdbid]
            prots_embedding.append(pro_embedding)

        self.smi = ligands
        self.pdbids = list(common_pdbids)
        self.prots = prots
        self.prots_embedding = prots_embedding
        self.smiles_embedding = smiless_embedding
        self.length = len(self.smi)
        self.max_seq_len = max_seq_len
        self.max_smi_len = max_smi_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pdbid = self.pdbids[idx]
        aug_smile = self.smi[pdbid]
        protseq = self.prots[pdbid]
        prots_embedding = torch.tensor(self.prots_embedding[idx])
        smiles_embedding = torch.tensor(self.smiles_embedding[idx])
        return  label_smiles(aug_smile, self.max_smi_len),label_sequence(protseq,self.max_seq_len),smiles_embedding, prots_embedding ,np.array(self.affinity[pdbid], dtype=np.float32)
