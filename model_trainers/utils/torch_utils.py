import torch
import torch.backends.cuda as cuda_back
# import torch.backends.mps as mps
import torch.cuda as cuda

from datasets import Dataset


def get_device() -> str:
    """returns the torch device to use for the current system"""

    if cuda.is_available() and cuda_back.is_built():
        return "cuda"
    # elif mps.is_available() and mps.is_built():
        # return "cpu"
    else:
        return "cpu"


def get_class_weights(ds: Dataset, label_list: list, min_value: int = 1000):

    count_list = [0] * len(label_list)

    def count_label(example):

        for token in example["verbnet"]:
            for role in token:
                count_list[role] += 1

    ds.map(count_label)

    for i, c in enumerate(count_list):
        if c < min_value:
            count_list[i] = min_value

    weight_tensor = torch.tensor(
        count_list, dtype=torch.float, device=get_device())

    return len(ds) / weight_tensor
