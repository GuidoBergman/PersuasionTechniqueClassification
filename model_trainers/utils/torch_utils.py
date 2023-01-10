import torch.backends.cuda as cuda_back
import torch.backends.mps as mps
import torch.cuda as cuda


def get_device() -> str:
    """returns the torch device to use for the current system"""

    if cuda.is_available() and cuda_back.is_built():
        return "cuda"
    elif mps.is_available() and mps.is_built():
        return "mps"
    else:
        return "cpu"
