import torch
import torch.backends.cuda as cuda_back
# import torch.backends.mps as mps
import torch.cuda as cuda

from datasets import Dataset
from typing import Union, List


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


def class_vector_to_multi_hot_vector(vec: Union[list, List[list]], num_classes: int) -> list:

    mh_vec = []
    if isinstance(vec[0], list):
        for sub_vec in vec:
            mh_vec.append(
                [1.0 if c in sub_vec else 0.0 for c in range(num_classes)])
    else:
        mh_vec = [1.0 if c in vec else 0.0 for c in range(num_classes)]
    return mh_vec


def multi_hot_vector_to_class_vector(vec: Union[list, List[list]]) -> list:
    if len(vec) == 0:
        return []
    
    class_vec = []
    if isinstance(vec[0], list):
        for sub_vec in vec:
            class_vec.append(
                [i for i, entry in enumerate(sub_vec) if entry > 0])
    else:
        class_vec = [i for i, entry in enumerate(vec) if entry > 0]
    return class_vec
