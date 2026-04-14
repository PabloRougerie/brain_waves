import glob
import torch
from pathlib import Path
import time
import numpy as np

from src.lightning import BrainLightning
from src.models import OptunaModel
from src.params import BEST_MODEL_PARAMS

def instantiate_model(path):
    
    #instantiate model with correct architecture
    best_model = OptunaModel(**BEST_MODEL_PARAMS)
    
    #load weights
    lit_model = BrainLightning.load_from_checkpoint(path, model= best_model)
    
    #get model
    model = lit_model.model
    
    return model 

    
def get_size_and_params(model):
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
    
        
    
def measure_latency(model, iterations):
    
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


def model_efficience_report(path= None, model= None, iterations=200):

    if not (bool(path) ^ bool(model)): 
        raise ValueError("provide either one path or one model, not both, not neither")
        
    if path:
        model = instantiate_model(path)
    
    
    n_total, n_trainable, size_all_mb = get_size_and_params(model)
    latencies = measure_latency(model, iterations)

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
    print(f"{sep}\n")
    
    return model
    
    
    

          
          
    
    