import torch
import pytorch_lightning as pl
from scportrait_ot.flow_matching import SourceConditionalFlowMatcher
from scportrait_ot.network import TimeConditionedMLP
from scportrait_ot.ode import ConditionalODE
from torchdiffeq import odeint

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
                 weight_decay: float = 1e-6, 
                 distance="euclidean"):
        
        super().__init__()

        # Store hyperparams
        self.save_hyperparameters()
        self.input_dim = input_dim
        
        # Initialize neural network
        self.v_mlp = TimeConditionedMLP(input_dim=input_dim, 
                                        hidden_dim=hidden_dim,
                                        num_hidden_layers=num_hidden_layers,
                                        source_condition_dim=source_condition_dim,
                                        time_embedding_dim=time_embedding_dim,
                                        use_batchnorm=use_batchnorm)
        
        # Initialize the Flow Matching framework
        self.fm = SourceConditionalFlowMatcher(sigma=sigma, 
                                               flavor=flavor,
                                               distance=distance)    
        
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
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """
        Training step for VDM.

        Args:
            batch: Batch data.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Loss value.
        """
        return self._step(batch, "valid")

    def _step(self, batch, phase):
        # Perform OT reordering 
        x0, _, t, xt, ut = self.fm.sample_location_and_conditional_flow(x0=batch["codex"],
                                                                        x1=batch["rna"], 
                                                                        x0_shared=batch["codex_shared"],
                                                                        x1_shared=batch["rna_shared"])
        
        # Evalauate flow matching model
        vt = self.v_mlp(xt, x0, t)

        # Evaluate the loss
        loss = self.criterion(ut, vt)
        
        # Save results
        metrics = {
            f"{phase}/loss": loss.mean()}
        self.log_dict(metrics, prog_bar=True)
        
        return loss.mean()

    def pushforward(
        self,
        x0: torch.Tensor,
        n_timesteps: int = 100,
        solver: str = "dopri5",
        atol: float = 1e-5,
        rtol: float = 1e-5
        ):
        """
        Solves the ODE from x0 over time using the learned flow.

        Args:
            x0 (torch.Tensor): Initial condition.
            n_timesteps (int): Number of time steps.
            solver (str): ODE solver to use (e.g., 'dopri5', 'rk4').
            atol (float): Absolute tolerance.
            rtol (float): Relative tolerance.
            method_kwargs (dict): Extra kwargs passed to the solver.
            t0 (float): Initial time.
            t1 (float): Final time.

        Returns:
            torch.Tensor: The trajectory of the solution.
        """
        time = torch.linspace(0, 1, n_timesteps, device=x0.device)
        node = ConditionalODE(self.v_mlp, x0)
        eps = torch.randn(x0.shape[0], self.input_dim)

        trajectory = odeint(
            node,
            eps,
            time,
            method=solver,
            atol=atol,
            rtol=rtol
        )
        return trajectory[-1]
        