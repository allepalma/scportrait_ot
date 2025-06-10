from dataloader import SingleCellAndCodexDataset 
from model import FlowMatchingModelWrapper
from torch.utils.data import random_split
import torch

class CfgenEstimator:
    def __init__(self, args):
        self.args =  args 
        self.init_datamodule()

    def init_datamodule(self):
        self.dataset = SingleCellAndCodexDataset(self.args.datamodule.rna_adata_path, 
                                                    self.args.datamodule.codex_adata_path, 
                                                    self.args.datamodule.shared_genes,
                                                    self.args.datamodule.label_columns, 
                                                    self.args.datamodule.obsm_key_rna, 
                                                    self.args.datamodule.obsm_key_codex)    
                                                    
        
        self.train_data, self.valid_data = random_split(self.dataset,
                                                        lengths=[0.80, 0.20])   
        
        self.train_dataloader = torch.utils.data.DataLoader(self.train_data,
                                                            batch_size=self.args.training_config.batch_size,
                                                            shuffle=True,
                                                            num_workers=4)
        
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_data,
                                                            batch_size=self.args.training_config.batch_size,
                                                            shuffle=False,
                                                            num_workers=4)
        
    def init_model(self):
        self.model = FlowMatchingModelWrapper(input_dim=self.dataset.input_dim,
                                                hidden_dim=self.args.model.hidden_dim,
                                                num_hidden_layers=self.args.model.num_hidden_layers,
                                                time_embedding_dim=self.args.model.time_embedding_dim,
                                                source_condition_dim=self.dataset.source_dim, 
                                                use_batchnorm=self.args.model.use_batchnorm,
                                                sigma=self.args.model.sigma, 
                                                flavor=self.args.model.flavor, 
                                                learning_rate=self.args.model.learning_rate, 
                                                weight_decay=self.args.model.weight_decay)


    def init_trainer(self):
        pass
            
    def init_feature_embeddings(self):
        pass

    def train(self):
        pass
    
    def test(self):
        pass
    