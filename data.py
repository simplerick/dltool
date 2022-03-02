import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


def get_batch(dataloader):
    try:
        return next(dataloader._iterator)
    except (TypeError, StopIteration):
        dataloader._iterator = dataloader.__iter__()
        return next(dataloader._iterator)


def with_transforms(cls):
    def _with_transforms(self, transforms):
        self.transforms = transforms
        return self

    def _getitem(self, *args, **kwargs):
        x = cls.__getitem__(self, *args, **kwargs)
        for t in self.transforms:
            x = t(x)
        return x

    class DatasetWithTransforms(cls):
        transforms = []
        with_transforms = _with_transforms
        __getitem__ = _getitem

    return DatasetWithTransforms


@with_transforms
class SequenceDataset(Dataset):
    def __init__(self, path, metadata=None, max_metadata_size=1024):
        self.file = Path(path)
        self._attrs = metadata.copy() if isinstance(metadata, dict) else {}
        if self.file.exists():
            with open(path, "rb") as f:
                self._read_attrs(f.read(max_metadata_size))

    def _read_attrs(self, byte_str):
        sep_ix = byte_str.find(b'\n')
        self._attrs.update(json.loads(byte_str[:sep_ix]))
        for key in self._attrs:
            setattr(self, key, self._attrs[key])
        self.offset = sep_ix + 1
        self.size = self.element_size * np.prod(self.shape)
        self.length = (self.file.stat().st_size - self.offset) // self.size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            start = index.start if (index.start is not None) else 0
            stop = index.stop if (index.stop is not None) else self.length
            start += (start < 0) * self.length
            stop += (stop < 0) * self.length
            start = min(self.length, max(0, start))
            stop = min(self.length, max(start, stop))
            if index.step:
                raise NotImplementedError("step in slice is not implemented")
        else:
            start = index + (index < 0) * self.length
            if start < 0 or start >= self.length:
                raise IndexError("index out of range")
            stop = index + 1
        with open(self.file, "rb") as f:
            f.seek(self.offset + self.size * start)
            byte_str = f.read((stop - start) * self.size)
        t = torch.from_numpy(np.frombuffer(byte_str, dtype=self.dtype)).view((stop - start), *self.shape)
        return t

    def write(self, tensors):
        if len(tensors) > 0:
            metadata = self.file.exists()
            file = open(self.file, "ab")
            for t in tensors:
                t_attrs = {'dtype': str(t.dtype).split(".")[1],
                           'element_size': t.element_size(),
                           'shape': list(t.size())}
                if not metadata:
                    byte_str = json.dumps({**self._attrs, **t_attrs}).encode("utf-8") + b'\n'
                    file.write(byte_str)
                    file.flush()
                    self._read_attrs(byte_str)
                    metadata = True
                if t_attrs.items() <= self._attrs.items():
                    file.write(t.numpy().tobytes())
                    self.length += 1
                else:
                    raise ValueError("tensor shape or dtype doesn't match with template")
            file.close()
