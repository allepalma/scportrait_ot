# Imports
import numpy as np 
import torch
from torch.utils.data import Dataset
import scanpy as sc
from sklearn.preprocessing import LabelEncoder

class SingleCellAndCodexDataset(Dataset):
    def __init__(self, 
                 rna_adata_path, 
                 codex_adata_path, 
                 label_columns, 
                 obsm_key_rna=None, 
                 obsm_key_codex=None):
        
        # Read datasets
        self.rna_adata = sc.read_h5ad(rna_adata_path)
        self.codex_adata = sc.read_h5ad(codex_adata_path)
        
        # Get the cell state to match 
        if obsm_key_rna:
            self.X_rna = self.rna_adata.obsm[obsm_key_rna]
        else:
            self.X_rna = self.rna_adata.X
        
        if obsm_key_codex:
            self.X_codex = self.codex_adata.obsm[obsm_key_codex]
        else:
            self.X_codex = self.codex_adata.X
            
        # Get shared cell states - will match in gene ids 
        self.X_rna_shared = self.rna_adata.X
        self.X_codex_shared = self.codex_adata.X

        # Input dim
        self.input_dim = self.X_rna.shape[1]
        self.source_dim = self.X_codex.shape[1]
        
        # Encode some columns 
        self.label_maps = {}
        self.encoded_labels = {}
        for column in label_columns:
            label_encoder = LabelEncoder()
            encoded = label_encoder.fit_transform(self.rna_adata.obs[column]).astype(float)
            self.encoded_labels[column] = encoded
            self.label_maps[column] = dict(enumerate(label_encoder.classes_))

    def __len__(self):
        return len(self.codex_adata)
    
    def _len_rna(self):
        return len(self.rna_adata)

    def __getitem__(self, idx):
        # Get observations and convert to float32
        X_codex_batch = torch.from_numpy(self.X_codex[idx]).float()
        X_codex_shared_batch = torch.from_numpy(self.X_codex_shared[idx]).float()

        idx_rna = np.random.choice(range(self._len_rna()))
        X_rna_batch = torch.from_numpy(self.X_rna[idx_rna]).float()
        X_rna_shared_batch = torch.from_numpy(self.X_rna_shared[idx_rna]).float()
        
        # Ensure labels are also float32
        encoded_labels = {
            key: torch.tensor(val[idx], dtype=torch.float32) if np.isscalar(val[idx]) 
                else torch.from_numpy(val[idx]).float()
            for key, val in self.encoded_labels.items()
        }

        return dict(
            codex=X_codex_batch, 
            rna=X_rna_batch,
            codex_shared=X_codex_shared_batch, 
            rna_shared=X_rna_shared_batch, 
            labels=encoded_labels
        )
        