# Imports
import numpy as np 
import pandas as pd 
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
import scanpy as sc
from sklearn.preprocessing import LabelEncoder

class EmbeddingDecoderDataset(Dataset):
    def __init__(self,
                 adata_path, 
                 count_label, 
                 embedding_label,
                 batch_label):

        super().__init__()

        # Load AnnData object
        self.adata = sc.read_h5ad(adata_path)
        
        # Convert count matrix and embeddings to PyTorch tensors
        if count_label is not None:
            self.X = torch.from_numpy(self.adata.layers[count_label].todense().astype('float32'))
        else:
            if sp.issparse(self.adata.X):
                self.X = torch.from_numpy(self.adata.X.todense().astype('float32'))
            else:
                self.X = torch.from_numpy(self.adata.X.astype('float32'))

        try: 
            self.X_emb = torch.from_numpy(self.adata.obsm[embedding_label].values.astype('float32'))
        except:
            self.X_emb = torch.from_numpy(self.adata.obsm[embedding_label].astype('float32'))
        
        self.input_dim = self.X_emb.shape[1]
        self.output_dim = self.X.shape[1]
        
        label_encoder = LabelEncoder()
        batch_data = self.adata.obs[batch_label].astype(str)  # ensure strings
        batch_encoded = label_encoder.fit_transform(batch_data)
        self.batch_data = torch.tensor(batch_encoded, dtype=torch.long)
        
        del self.adata
        
        # One-hot encode batch labels
        num_classes = len(label_encoder.classes_)
        self.batch_data = torch.nn.functional.one_hot(self.batch_data, num_classes=num_classes).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        batch_dict = {
            "X": self.X[idx],
            "X_emb": self.X_emb[idx],
            "batch_one_hot": self.batch_data[idx]
        }
        return batch_dict
    
class SingleCellAndCodexDataset(Dataset):
    def __init__(self, 
                 rna_adata_path, 
                 codex_adata_path, 
                 label_columns, 
                 obsm_key_rna=None, 
                 obsm_key_codex=None, 
                 rna_sampling_label=None, 
                 uniform_sampling_rna=False):
        
        # Read datasets
        self.rna_adata = sc.read_h5ad(rna_adata_path)
        self.codex_adata = sc.read_h5ad(codex_adata_path)
        
        # Get the cell state to match 
        if obsm_key_rna:
            obsm_rna = self.rna_adata.obsm[obsm_key_rna]
            if isinstance(obsm_rna, pd.DataFrame):
                self.X_rna = obsm_rna.values
            else:
                self.X_rna = obsm_rna
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
        
        # Initialize uniform sampling variable
        if uniform_sampling_rna: 
            assert rna_sampling_label is not None, "You must provide an RNA sampling label with uniform sampling as a column of .obs"
        
        self.uniform_sampling_rna = uniform_sampling_rna
        self.rna_sampling_label = rna_sampling_label
        if self.rna_sampling_label and self.uniform_sampling_rna:
            self.labels = self.rna_adata.obs[self.rna_sampling_label].values
            self.unique_labels = np.unique(self.labels)

    def __len__(self):
        return len(self.codex_adata)
    
    def _len_rna(self):
        return len(self.rna_adata)
    
    def _sample_uniform_from_label(self):
        # Ensure label exists
        if self.rna_sampling_label not in self.rna_adata.obs.columns:
            raise ValueError(f"Label '{self.rna_sampling_label}' not found in rna_adata.obs")

        # Sample a label uniformly
        sampled_label = self.unique_labels[np.random.randint(len(self.unique_labels))]

        # Get indices corresponding to that label
        matching_indices = np.where(self.labels == sampled_label)[0]
        
        # # Uniformly sample one index from those
        return matching_indices[np.random.randint(len(matching_indices))]
        
    def __getitem__(self, idx):
        # Get observations and convert to float32
        X_codex_batch = torch.from_numpy(self.X_codex[idx]).float()
        X_codex_shared_batch = torch.from_numpy(self.X_codex_shared[idx]).float()

        if self.uniform_sampling_rna:
            idx_rna = self._sample_uniform_from_label()
        else:
            idx_rna = np.random.randint(self._len_rna())

        X_rna_batch = torch.from_numpy(self.X_rna[idx_rna]).float()
        X_rna_shared_batch = torch.from_numpy(self.X_rna_shared[idx_rna]).float()
        
        # Ensure labels are also float32
        encoded_labels = {
            key: torch.tensor(val[idx_rna], dtype=torch.float32) if np.isscalar(val[idx]) 
                else torch.from_numpy(val[idx_rna]).float()
            for key, val in self.encoded_labels.items()
        }

        return dict(
            codex=X_codex_batch, 
            rna=X_rna_batch,
            codex_shared=X_codex_shared_batch, 
            rna_shared=X_rna_shared_batch, 
            labels=encoded_labels
        )
        