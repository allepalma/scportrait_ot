import torch
import pytorch_lightning as pl
from flow_matching import SourceConditionalFlowMatcher
from network import TimeConditionedMLP

class FlowMatchingModelWrapper(pl.LightningModule):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_hidden_layers: int,
                 time_embedding_dim: int,
                 source_condition_dim: int, 
                 use_batchnorm: bool = False,
                 sigma: float = 0, 
                 flavor: str = "cfm", 
                 learning_rate: float = 1e-4, 
                 weight_decay: float = 1e-6):
        
        super().__init__()

        # Store hyperparams
        self.save_hyperparameters()
        
        # Initialize neural network
        self.v_mlp = TimeConditionedMLP(input_dim=input_dim, 
                                        hidden_dim=hidden_dim,
                                        num_hidden_layers=num_hidden_layers,
                                        time_embedding_dim=time_embedding_dim,
                                        source_condition_dim=source_condition_dim,
                                        use_batchnorm=use_batchnorm)
        
        # Initialize the Flow Matching framework
        self.fm = SourceConditionalFlowMatcher(sigma=sigma, 
                                               flavor=flavor)    
        
        # Other parameters 
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # MSE lost for the Flow Matching algorithm 
        self.criterion = torch.nn.MSELoss()
        
    def configure_optimizers(self):
        """Initialize optimizer
        """
        params = list(self.parameters())
        optimizer = torch.optim.AdamW(params, 
                                        self.learning_rate, 
                                        weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        Training step for VDM.

        Args:
            batch: Batch data.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        return self._step(batch)

    def validation_step(self, batch, batch_idx):
        """
        Training step for VDM.

        Args:
            batch: Batch data.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        return self._step(batch)

    def _step(self, batch):
        # Perform OT reordering 
        x0, _, x0_shared, _, t, xt, ut = self.fm.sample_location_and_conditional_flow(x0=batch["codex"],
                                                                                        x1=batch["rna"], 
                                                                                        x0_shared=batch["codex_shared"],
                                                                                        x1_shared=batch["rna_shared"])
        
        # Evalauate flow matching model
        vt = self.v_mlp(xt, x0, t)

        # Evaluate the loss
        loss = self.criterion(ut, vt)
        
        # Save results
        metrics = {
            f"{loss}": loss.mean()}
        self.log_dict(metrics, prog_bar=True)
        
        return loss.mean()
    