from collections.abc import Sequence
from torch.utils.data import Dataset
from .utils import transformable


@transformable
class UnionDataset(Dataset):
    __slots__ = ["datasets"]

    def __init__(self, datasets: Sequence[Dataset]):
        self.datasets = datasets

    def __len__(self):
        return min([len(d) for d in self.datasets])

    def __getitem__(self, index: int):
        return tuple(d[index] for d in self.datasets)
