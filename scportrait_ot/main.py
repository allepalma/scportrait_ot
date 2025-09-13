import hydra
import sys
from omegaconf import DictConfig
from scportrait_ot.experiment import FlowMatchingExperiment

@hydra.main(config_path="../configs", config_name="train", version_base=None)
def train(cfg: DictConfig):
    """
    Main training function using Hydra.

    Args:
        cfg (DictConfig): Configuration parameters.

    Raises:
        Exception: Any exception during training.

    Returns:
        None
    """
    # Initialize estimator 
    estimator = FlowMatchingExperiment(cfg)
    # Train and test 
    estimator.train()
    estimator.test()
    return None
    
if __name__ == "__main__":
    import traceback
    try:
        train()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise
    