import warnings
from multiprocessing import Process
from typing import Any, Union, Sequence

import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


def get_batch(dataloader: DataLoader) -> Any:
    try:
        return next(dataloader._batch_iter)
    except (AttributeError, TypeError, StopIteration):
        dataloader._batch_iter = dataloader.__iter__()
        return next(dataloader._batch_iter)


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
    def __init__(self, path: Union[str, Path], metadata: dict = None, max_metadata_size: int = 1024):
        self.file = Path(path)
        self._attrs = metadata.copy() if isinstance(metadata, dict) else {}
        if self.file.exists():
            with open(path, "rb") as f:
                self._read_attrs(f.read(max_metadata_size))
        else:
            warnings.warn(f"path '{path}' does not exist, it will be interpreted as the desired storage location.")
            self.length = 0
            self.size = 0

    def _read_attrs(self, byte_str: bytes):
        sep_ix = byte_str.find(b'\n')
        self._attrs.update(json.loads(byte_str[:sep_ix]))
        for key in self._attrs:
            setattr(self, key, self._attrs[key])
        self.offset = sep_ix + 1
        self.size = int(self.element_size * np.prod(self.shape))
        self.length = (self.file.stat().st_size - self.offset) // self.size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
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

    @staticmethod
    def tensor_attrs(t: torch.Tensor) -> dict:
        return {'dtype': str(t.dtype).split(".")[1],
                'element_size': t.element_size(),
                'shape': list(t.size())}

    @staticmethod
    def _write(file_path: Path, chunk: Sequence[torch.Tensor], offset: int):
        # it is more efficient to concatenate first
        if not isinstance(chunk, torch.Tensor):
            chunk = torch.cat(chunk, 0)
        byte_str = chunk.cpu().numpy().tobytes()
        with open(file_path, "r+b") as f:
            f.seek(offset)
            f.write(byte_str)
            f.flush()

    def write(self, sequence: Sequence[torch.Tensor], processes: int = 1):
        if len(sequence) > 0:
            metadata = self.file.exists()
            if not metadata:
                t_attrs = self.tensor_attrs(sequence[0])
                byte_str = json.dumps({**self._attrs, **t_attrs}).encode("utf-8") + b'\n'
                with open(self.file, "ab") as file:
                    file.write(byte_str)
                    file.flush()
                self._read_attrs(byte_str)

            procs = []
            chunk_size = (len(sequence) + processes - 1) // processes
            end_of_file = self.offset + self.length * self.size
            for ix in range(0, len(sequence), chunk_size):
                chunk = sequence[ix: ix + chunk_size]
                offset = end_of_file + self.size * ix
                p = Process(target=self._write, args=(self.file, chunk, offset))
                procs.append(p)
                p.start()
            for p in procs:
                p.join()
            self.length += len(sequence)
