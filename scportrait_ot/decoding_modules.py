import torch 
import torch.nn.functional as F
import pytorch_lightning as pl
from scvi.distributions import NegativeBinomial
from scportrait_ot.network import MLP

class DecoderFromHarmony(pl.LightningModule):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 dims,
                 batch_norm, 
                 dropout,
                 dropout_p, 
                 batch_encoding, 
                 batch_encoding_dim, 
                 learning_rate,
                 weight_decay=1e-6,
                 likelihood: str = "nb"  # <-- NEW ARG
                ):
        super().__init__()

        # Initialize the attributes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dims = dims
        self.batch_norm = batch_norm
        self.dropout = dropout 
        self.dropout_p = dropout_p
        self.batch_encoding = batch_encoding 
        self.batch_encoding_dim = batch_encoding_dim
        self.learning_rate = learning_rate 
        self.weight_decay = weight_decay
        assert likelihood in ["nb", "gaussian"], \
            f"likelihood must be 'nb' or 'gaussian', got {likelihood}"
        self.likelihood = likelihood

        if batch_encoding: 
            self.input_dim = self.input_dim + self.batch_encoding_dim
        
        # Initialize theta (only for NB decoding)
        if self.likelihood == "nb":
            self.theta = torch.nn.Parameter(torch.randn(output_dim), 
                                            requires_grad=True)

        # Decoder 
        layer_dims = [input_dim] + self.dims + [output_dim]
        self.decoder = MLP(dims=layer_dims,
                           batch_norm=self.batch_norm,
                           dropout=self.dropout,
                           dropout_p=self.dropout_p)


    def _step(self, batch, dataset_type):
        # Collect from the batch
        x = batch["X_emb"]
        y = batch["X"]
        batch = batch["batch_one_hot"] 
        size_factor = y.sum(1, keepdim=True)
        if self.batch_encoding:
            x = torch.cat([x, batch], dim=1)

        # Decode 
        y_hat = self.decoder(x)

        if self.likelihood == "gaussian":
            # --- Gaussian decoding with MSE ---
            loss = F.mse_loss(y_hat, y, reduction="mean")
        else:  # "nb"
            # --- Negative Binomial decoding ---
            mu_hat = F.softmax(y_hat, dim=1)
            px = NegativeBinomial(mu=mu_hat * size_factor, theta=torch.exp(self.theta))
            loss = -px.log_prob(y).sum(1).mean()

        self.log(f'{dataset_type}/loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "valid")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(),
                                 lr=self.learning_rate,
                                 weight_decay=self.weight_decay)
        