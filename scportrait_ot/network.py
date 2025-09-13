import torch
import torch.nn as nn
from typing import List, Optional, Callable
import math

def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class MLP(nn.Module):
    def __init__(
        self,
        dims: List[int],
        batch_norm: bool,
        dropout: bool,
        dropout_p: float,
        activation: Optional[Callable] = nn.ELU,
        final_activation: Optional[str] = None,
    ):
        super().__init__()
        # Attribute initialization
        self.dims = dims
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.activation = activation
        self.final_activation_type = final_activation

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(activation())
            if dropout:
                layers.append(nn.Dropout(dropout_p))

        # Final layer (no activation)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

        if final_activation == "tanh":
            self.final_activation = nn.Tanh()
        elif final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch_size, dims[0])
        """
        x = self.net(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x

class TimeConditionedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        source_condition_dim: int,
        time_embedding_dim: int,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        # Attribute initialization
        self.time_embedding_dim = time_embedding_dim
        self.source_condition_dim = source_condition_dim
        self.use_batchnorm = use_batchnorm
        act_fn = nn.SELU() if not use_batchnorm else nn.ELU()

        layers = []
        in_dim = input_dim + source_condition_dim + time_embedding_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn)
            in_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor,  x0: torch.Tensor, t: torch.Tensor):
        """
        x: shape (batch_size, input_dim)
        t: shape (batch_size,) or (batch_size, 1)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t_embed = timestep_embedding(t, self.time_embedding_dim)
        x_input = torch.cat([x, x0, t_embed], dim=-1)
        h = self.hidden(x_input)
        return self.output(h)
    