import glob
import torch
from pathlib import Path
import time
import numpy as np

from src.lightning import BrainLightning
from src.models import OptunaModel
from src.params import BEST_MODEL_PARAMS

def instantiate_model(path):
    """Load a BrainLightning checkpoint and return the inner OptunaModel on CPU.
    
    Args:
        path: Path to the .ckpt checkpoint file.
    
    Returns:
        OptunaModel with loaded weights, in eval-ready state on CPU.
    """
    #instantiate model with correct architecture
    best_model = OptunaModel(**BEST_MODEL_PARAMS)
    
    #load weights
    lit_model = BrainLightning.load_from_checkpoint(path, model= best_model)
    
    #get model
    model = lit_model.model
    model.cpu()
    
    return model 

    
def get_size_and_params(model):
    """Compute parameter counts and in-memory size of a float model.

    Args:
        model: A torch.nn.Module (non-quantized).

    Returns:
        Tuple (n_total, n_trainable, size_mb):
            n_total      -- total number of parameters
            n_trainable  -- number of trainable parameters
            size_mb      -- model size in MB (parameters + buffers)
    """
    param_size = 0
    buffer_size = 0
    n_trainable = 0
    n_total = 0
    
    
    for param in model.parameters():
        n_total += param.numel()
        n_trainable += param.numel() if param.requires_grad else 0
        param_size += param.numel() * param.element_size()
        
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
        
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    
    return n_total, n_trainable, size_all_mb


def get_size_quantization(model):
    """Measure the on-disk size of a quantized model by saving its state_dict.

    Uses /tmp as a temporary write location. More accurate than counting
    parameter bytes for quantized models since dtypes vary per layer.

    Args:
        model: A quantized torch.nn.Module.

    Returns:
        File size in MB.
    """
    torch.save(model.state_dict(), "/tmp/quant_model.pt") 
    size_quant_mb = Path("/tmp/quant_model.pt").stat().st_size / 1024**2
    return size_quant_mb
    
    
        
    
def measure_latency(model, iterations):
    """Measure per-inference latency on CPU with a dummy input.

    Runs 10 warmup iterations before recording, to avoid cold-start bias.

    Args:
        model:      A torch.nn.Module (float or quantized).
        iterations: Number of timed inference runs.

    Returns:
        List of latencies in seconds (length = iterations).
    """
    #get to eval model
    model.eval()
    
    #assign to cpu
    model.cpu()
    
    #dummy input tensor of shape (batch, channels, freqs, time)
    dummy = torch.randn(1,4,100,25)
    
    with torch.no_grad():
        for _ in range(10):
            model(dummy)

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            model(dummy)
            latencies.append(time.perf_counter() - t0)

    return latencies


def evaluate_model(model, datamodule):
    """Compute mean KL divergence loss on the validation set.

    Args:
        model:      A torch.nn.Module producing log-probabilities.
        datamodule: A BrainDataModule with a configured val_dataloader.

    Returns:
        Mean KLDivLoss (float) over the full validation set.
    """
    model.eval()
    model.cpu()
    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    
    losses = []
    with torch.no_grad():
        for x, y in datamodule.val_dataloader():
            x, y = x.cpu(), y.cpu()
            logits = model(x)
            loss = criterion(logits,y)
            losses.append(loss)
            
    return np.mean(losses)
            


def model_efficience_report(path=None, model=None, datamodule=None, iterations=5000):
    """Print and return efficiency metrics for a standard (non-quantized) model.

    Exactly one of `path` or `model` must be provided.

    Args:
        path:       Path to a .ckpt checkpoint file (optional).
        model:      An already-instantiated OptunaModel (optional).
        datamodule: BrainDataModule for KL score evaluation (optional).
        iterations: Number of latency measurement iterations (default 5000).

    Returns:
        Tuple (size_mb, median_latency_ms, score) where score is None if
        no datamodule was provided.
    """
    print("load model")
    if (path is None) == (model is None): 
        raise ValueError("provide either one path or one model, not both, not neither")
        
    if path:
        model = instantiate_model(path)
    
    print("calculate size and latency")
    n_total, n_trainable, size_all_mb = get_size_and_params(model)
    latencies = measure_latency(model, iterations)
    
    if datamodule:
        print("calculate model inference score")
        score = evaluate_model(model, datamodule)
    else:
        score = None

    sep = "-" * 35
    print(f"\n{sep}")
    print(f"  Model Efficiency Report")
    print(sep)
    print(f"  Total params:        {n_total:,}")
    print(f"  Trainable params:    {n_trainable:,}")
    print(f"  Model size (MB):     {size_all_mb:.3f}")
    print(sep)
    print(f"  Median latency (ms): {np.median(latencies) * 1000:.2f}")
    print(f"  Max latency (ms):    {np.max(latencies) * 1000:.2f}")
    print(sep)
    print(f"  KL score: {score:.4f}" if score is not None else "  KL score: N/A")
    print(f"{sep}\n")
    
    return size_all_mb, np.median(latencies)*1000, score
    
    
    
def quantized_model_efficience_report(model, datamodule=None, iterations=5000):
    """Print and return efficiency metrics for a quantized model.

    Size is measured via on-disk serialization to account for mixed dtypes.

    Args:
        model:      A quantized torch.nn.Module (e.g. output of convert()).
        datamodule: BrainDataModule for KL score evaluation (optional).
        iterations: Number of latency measurement iterations (default 5000).

    Returns:
        Tuple (size_mb, median_latency_ms, score) where score is None if
        no datamodule was provided.
    """
    size = get_size_quantization(model)
    latencies = measure_latency(model, iterations)
    if datamodule:
        print("calculate model inference score")
        score = evaluate_model(model, datamodule)
    else: 
        score = None
    
    sep = "-" * 35
    print(f"\n{sep}")
    print(f"  Model Efficiency Report Quantization")
    print(sep)
    print(f"  Model size (MB):     {size:.3f}")
    print(sep)
    print(f"  Median latency (ms): {np.median(latencies) * 1000:.2f}")
    print(f"  Max latency (ms):    {np.max(latencies) * 1000:.2f}")
    print(sep)
    print(f"  KL score: {score:.4f}" if score is not None else "  KL score: N/A")
    print(f"{sep}\n")
    
    return size, np.median(latencies)*1000, score
    
    

          
          
    
    