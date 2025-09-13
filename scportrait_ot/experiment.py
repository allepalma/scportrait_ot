from pathlib import Path
from scportrait_ot.dataloader import SingleCellAndCodexDataset 
from scportrait_ot.model import FlowMatchingModelWrapper
from torch.utils.data import random_split
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

class FlowMatchingExperiment:
    def __init__(self, args):
        self.args =  args 
        self.init_datamodule()
        self.init_model()
        self.init_trainer()

    def init_datamodule(self):
        self.dataset = SingleCellAndCodexDataset(self.args.datamodule.rna_adata_path, 
                                                    self.args.datamodule.codex_adata_path, 
                                                    self.args.datamodule.label_columns, 
                                                    self.args.datamodule.obsm_key_rna, 
                                                    self.args.datamodule.obsm_key_codex, 
                                                    self.args.datamodule.rna_sampling_label, 
                                                    self.args.datamodule.uniform_sampling_rna)    
                                                    
        
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
                                                weight_decay=self.args.model.weight_decay, 
                                                distance=self.args.model.distance)


    def init_trainer(self):
        """
        Initialize Trainer
        """
        # Initialize WandbLogger
        self.logger = WandbLogger(save_dir=self.args.training_config.training_dir,
                                **self.args.logger)

        # Use wandb run name to create a subfolder
        run_name = self.logger.experiment.name
        run_dir = Path(self.args.training_config.training_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Callbacks for saving checkpoints in the run-specific folder
        checkpoint_callback = ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            **self.args.checkpoints
        )
        callbacks = [checkpoint_callback]

        # Initialize trainer with custom dir
        self.trainer_generative = Trainer(
            callbacks=callbacks,
            default_root_dir=run_dir,
            logger=self.logger,
            **self.args.trainer
        )
    
    def train(self):
        """
        Train the generative model using the provided trainer.
        """
        self.trainer_generative.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.valid_dataloader)
    
    def test(self):
        """
        Test the generative model.
        """
        self.trainer_generative.test(
            self.model,
            dataloaders=self.valid_dataloader)
    