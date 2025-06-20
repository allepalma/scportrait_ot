import torch 

class ConditionalODE(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""
    def __init__(self, model, x0):
        
        super().__init__()
        
        self.model = model  # The model being wrapped
        self.x0 = x0  # Conditioning variable

    def forward(self, t, x, *args, **kwargs):
        """
        Forward pass of the torch_wrapper.

        Args:
            t: Time tensor, will be repeated for each sample in the batch.
            x: Input tensor.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The output of the model after applying conditioning.
        """
        # Repeat and concatenate time tensor to match the batch size
        t = t.repeat(x.shape[0])[:, None]

        # Evaluate velocity
        vt = self.model(x, self.x0, t)
        return vt
    