# Standard library
import time
from pathlib import Path

# Third-party
import numpy as np
import torch
import torch.nn as nn
from onnxruntime.quantization import CalibrationDataReader, quantize_static  # noqa: F401

# Local
from src.lightning import BrainLightning
from src.models import OptunaModel
from src.params import BEST_MODEL_PARAMS


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def instantiate_model(path):
    """Load a BrainLightning checkpoint and return the inner OptunaModel on CPU.

    Args:
        path: Path to the .ckpt checkpoint file.

    Returns:
        OptunaModel with loaded weights, in eval-ready state on CPU.
    """
    best_model = OptunaModel(**BEST_MODEL_PARAMS)
    lit_model = BrainLightning.load_from_checkpoint(path, model=best_model)
    model = lit_model.model
    model.cpu()
    return model


def get_ignored_layers(model):
    """Return the OptunaModel layers to exclude from pruning.

    Args:
        model: An instantiated OptunaModel.

    Returns:
        List of nn.Module instances to pass as ignored_layers to MagnitudePruner.
    """
    return [model.backbone[0][0].block[0], model.head]


# ─────────────────────────────────────────────────────────────────────────────
# Measurement primitives
# ─────────────────────────────────────────────────────────────────────────────

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
    return Path("/tmp/quant_model.pt").stat().st_size / 1024**2


def measure_latency(model, iterations):
    """Measure per-inference latency on CPU with a dummy input.

    Runs 10 warmup iterations before recording, to avoid cold-start bias.

    Args:
        model:      A torch.nn.Module (float or quantized).
        iterations: Number of timed inference runs.

    Returns:
        List of latencies in seconds (length = iterations).
    """
    model.eval()
    model.cpu()
    dummy = torch.randn(1, 4, 100, 25)

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
            loss = criterion(logits, y)
            losses.append(loss)

    return np.mean(losses)


# ─────────────────────────────────────────────────────────────────────────────
# Standard model efficiency reports
# ─────────────────────────────────────────────────────────────────────────────

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

    return size_all_mb, np.median(latencies) * 1000, score


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

    return size, np.median(latencies) * 1000, score


# ─────────────────────────────────────────────────────────────────────────────
# Quantization helpers
# ─────────────────────────────────────────────────────────────────────────────

class QuantWrapperCustom(nn.Module):
    """Wraps an OptunaModel with quantization stubs for static PTQ.

    Places QuantStub before the backbone and DeQuantStub after the head,
    then applies log_softmax in float outside the quantized graph to avoid
    unsupported quantized ops on CPU.
    """

    def __init__(self, model):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.model = model
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model.backbone(x)
        x = self.model.head(x)
        x = self.dequant(x)
        return nn.functional.log_softmax(x, dim=1)


class BrainCalibrationReader(CalibrationDataReader):
    """Calibration data reader for ONNX static quantization.

    Iterates over the validation dataloader and stores batches as numpy
    arrays in the format expected by onnxruntime's quantize_static.

    Args:
        datamodule:  A BrainDataModule with a configured val_dataloader.
        max_batches: Maximum number of calibration batches (default 10).
    """

    def __init__(self, datamodule, max_batches=10):
        super().__init__()
        self.data = []

        for i, (x, y) in enumerate(datamodule.val_dataloader()):
            self.data.append(x.numpy().astype(np.float32))
            if i >= max_batches:
                break

        self.iter = iter(self.data)

    def get_next(self):
        try:
            return {"spectrogram": next(self.iter)}
        except StopIteration:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# ONNX utilities
# ─────────────────────────────────────────────────────────────────────────────

def onnx_size_mb(path):
    """Return the total size of an ONNX model in MB, including external data files.

    The dynamo-based ONNX exporter may store weights in sibling files alongside
    the .onnx graph file. This function accounts for all of them.

    Args:
        path: Path to the .onnx file.

    Returns:
        Combined file size in MB.
    """
    path = Path(path)
    total = path.stat().st_size
    for f in path.parent.glob(path.stem + "*"):
        if f != path:
            total += f.stat().st_size
    return total / 1024**2


def onnx_latency(session, iterations=5000):
    """Measure per-inference latency of an ONNX session with a dummy input.

    Runs 10 warmup iterations before recording, to avoid cold-start bias.

    Args:
        session:    An onnxruntime.InferenceSession.
        iterations: Number of timed inference runs.

    Returns:
        Tuple (median_latency_ms, max_latency_ms).
    """
    dummy = np.random.randn(1, 4, 100, 25).astype(np.float32)

    for _ in range(10):
        session.run(["class_logits"], {"spectrogram": dummy})

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(["class_logits"], {"spectrogram": dummy})
        latencies.append(time.perf_counter() - t0)

    return np.median(latencies) * 1000, np.max(latencies) * 1000


def onnx_kl_score(session, datamodule):
    """Compute mean KL divergence loss on the validation set via ONNX inference.

    Args:
        session:    An onnxruntime.InferenceSession producing log-probabilities.
        datamodule: A BrainDataModule with a configured val_dataloader.

    Returns:
        Mean KLDivLoss (float) over the full validation set.
    """
    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    losses = []

    for x, y in datamodule.val_dataloader():
        x_np = x.numpy().astype(np.float32)
        logits = session.run(["class_logits"], {"spectrogram": x_np})
        logits_t = torch.tensor(logits[0])  # session.run returns a list of arrays
        loss = criterion(logits_t, y).item()
        losses.append(loss)

    return np.mean(losses)


def onnx_model_efficience_report(path, session, datamodule=None, iterations=5000):
    """Print and return efficiency metrics for an ONNX model.

    Args:
        path:       Path to the .onnx file (for size measurement).
        session:    An onnxruntime.InferenceSession for the same model.
        datamodule: BrainDataModule for KL score evaluation (optional).
        iterations: Number of latency measurement iterations (default 5000).

    Returns:
        Tuple (size_mb, median_latency_ms, score) where score is None if
        no datamodule was provided.
    """
    size = onnx_size_mb(path)
    latency_median, latency_max = onnx_latency(session, iterations)

    if datamodule:
        score = onnx_kl_score(session, datamodule)
    else:
        score = None

    sep = "-" * 35
    print(f"\n{sep}")
    print(f"  Model Efficiency Report ONNX")
    print(sep)
    print(f"  Model size (MB):     {size:.3f}")
    print(sep)
    print(f"  Median latency (ms): {latency_median:.2f}")
    print(f"  Max latency (ms):    {latency_max:.2f}")
    print(sep)
    print(f"  KL score: {score:.4f}" if score is not None else "  KL score: N/A")
    print(f"{sep}\n")

    return size, latency_median, score
