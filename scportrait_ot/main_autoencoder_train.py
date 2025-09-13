import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import random_split
from scportrait_ot.dataloader import EmbeddingDecoderDataset
from scportrait_ot.decoding_modules import DecoderFromHarmony
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import sys
from pathlib import Path

@hydra.main(config_path="../configs_autoencoder", config_name="train", version_base=None)
def train(cfg: DictConfig):
    # Initialize dataset 
    dataset = EmbeddingDecoderDataset(adata_path=cfg.adata_path, 
                                        count_label=cfg.count_label, 
                                        embedding_label=cfg.embedding_label,
                                        batch_label=cfg.batch_label)
    
    # Fix the data 
    train_data, valid_data = random_split(dataset,
                                          lengths=[0.80, 0.20])   
        
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=cfg.batch_size,
                                                    shuffle=True,
                                                    num_workers=4)
            
    valid_dataloader = torch.utils.data.DataLoader(valid_data,
                                                    batch_size=cfg.batch_size,
                                                    shuffle=False,
                                                    num_workers=4)

    # Initialize decoder     
    decoder_model = DecoderFromHarmony(input_dim=dataset.input_dim, 
                                        output_dim=dataset.output_dim,
                                        dims=cfg.dims,
                                        batch_norm=cfg.batch_norm, 
                                        dropout=cfg.dropout,
                                        dropout_p=cfg.dropout_p, 
                                        batch_encoding=cfg.batch_encoding, 
                                        batch_encoding_dim=cfg.batch_encoding_dim,
                                        learning_rate=cfg.learning_rate,
                                        likelihood=cfg.likelihood
                                        )
    
    # Prepare training
    training_dir = Path(cfg.training_dir)

    logger = WandbLogger(offline=False,
                            anonymous=None,
                            project=cfg.project,
                            log_model=False,
                            save_dir=training_dir
                            )

    # Use wandb run name to create a subfolder
    run_name = logger.experiment.name
    run_dir = training_dir / cfg.project
    run_dir =  run_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks for saving checkpoints in the run-specific folder
    checkpoint_callback = ModelCheckpoint(dirpath=run_dir / "checkpoints",
                                            filename="epoch_{epoch:01d}",
                                            monitor="valid/loss",
                                            mode="min",               
                                            every_n_epochs=50,
                                            save_last=True,
                                            auto_insert_metric_name=False
                                            )
    callbacks = [checkpoint_callback]

    # Initialize trainer with custom dir
    trainer = Trainer(
        callbacks=callbacks,
        default_root_dir=run_dir,
        logger=logger,
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        devices=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        detect_anomaly=True,
        deterministic=False,
        gradient_clip_val=1)
    
    # Train model 
    trainer.fit(decoder_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloader)
        
if __name__ == "__main__":
    import traceback
    try:
        train()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise
    