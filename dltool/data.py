import warnings
from multiprocessing import Process
from typing import Any, Union, Sequence, Callable
from copy import copy, deepcopy

import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class DataIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def __next__(self):
        try:
            return next(self._iter)
        except (AttributeError, TypeError, StopIteration):
            self._iter = iter(self.dataloader)
            return next(self._iter)


def transformable(cls: type):
    """
    Modify the class to be able to transform the data. Warning: it modifies the class in place.

    Args:
        cls: initial class

    Returns:
        modified class
    """
    if hasattr(cls, "_getitem"):
        raise ValueError("The class already has `_getitem` method. Check if it is already transformable.")

    def _with_transforms(self, transforms: Sequence[Callable]):
        self.transforms = transforms
        return self

    cls.with_transforms = _with_transforms
    cls._getitem = cls.__getitem__

    def _getitem(self, *args, **kwargs):
        x = cls._getitem(self, *args, **kwargs)
        trfm = getattr(self, "transforms", [])
        for t in trfm:
            x = t(x)
        return x

    cls.__getitem__ = _getitem
    return cls


@transformable
class SequenceDataset(Dataset):
    _required_attrs = {'dtype', 'dtype_size', 'shape'}
    __slots__ = ["_file", "_attrs", "_element_size", "_length", "_offset"] + [f"_{a}" for a in _required_attrs]

    def __init__(self,
                 path: Union[str, Path],
                 metadata: dict = None,
                 overwrite: bool = False,
                 max_metadata_size: int = 1024):
        self._file = Path(path)
        self._attrs = metadata.copy() if isinstance(metadata, dict) else {}
        self._element_size = 0
        self._length = 0
        self._offset = 0
        if metadata is None:
            if self._file.exists():
                with open(path, "rb") as f:
                    self._read_attrs(f.read(max_metadata_size))
            else:
                raise RuntimeError(f"File {path} does not exist.")
        else:
            if self._file.exists():
                if overwrite:
                    warnings.warn(f"File {path} already exists and will be overwritten")
                else:
                    raise RuntimeError(f"File {path} already exists. Use overwrite=True to overwrite it")

    @property
    def file(self):
        return str(self._file)

    @property
    def metadata(self):
        return deepcopy(self._attrs)

    def _read_attrs(self, byte_str: bytes):
        try:
            sep_ix = byte_str.find(b'\n')
            self._attrs.update(json.loads(byte_str[:sep_ix]))
            if not self._attrs.keys() >= SequenceDataset._required_attrs:
                raise KeyError
            for key in self._attrs:
                setattr(self, f"_{key}", self._attrs[key])
            self._element_size = int(self._dtype_size * np.prod(self._shape))
            self._length = (self._file.stat().st_size - sep_ix - 1) // self._element_size
            self._offset = sep_ix + 1
        except:
            warnings.warn(f"reading metadata for {self.file} has failed. "
                          f"If writing is attempted, the file will be overwritten.")

    def _write_metadata(self, metadata: dict):
        if self._file.exists():
            self._file.unlink()
        byte_str = json.dumps(metadata).encode("utf-8") + b'\n'
        with open(self._file, "wb") as file:
            file.write(byte_str)
            file.flush()
        self._read_attrs(byte_str)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> torch.Tensor:
        if isinstance(index, slice):
            start = index.start if (index.start is not None) else 0
            stop = index.stop if (index.stop is not None) else self._length
            start += (start < 0) * self._length
            stop += (stop < 0) * self._length
            start = min(self._length, max(0, start))
            stop = min(self._length, max(start, stop))
            if index.step:
                raise NotImplementedError("step in slice is not implemented")
        else:
            start = index + (index < 0) * self._length
            if start < 0 or start >= self._length:
                raise IndexError("index out of range")
            stop = index + 1
        if stop == start:
            return torch.empty(0, *getattr(self, "_shape", []))
        with open(self._file, "rb") as f:
            f.seek(self._offset + self._element_size * start)
            byte_str = f.read((stop - start) * self._element_size)
        t = torch.from_numpy(np.frombuffer(byte_str, dtype=self._dtype)).view((stop - start), *self._shape)
        return t

    @staticmethod
    def _tensor_attrs(t: torch.Tensor) -> dict:
        return {'dtype': str(t.dtype).split(".")[1],
                'dtype_size': t.element_size(),
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

    def write(self, sequence: Sequence[torch.Tensor], index: int = None, processes: int = 1):
        if len(sequence) > 0:
            if self._offset == 0:
                t_attrs = self._tensor_attrs(sequence[0])
                self._write_metadata({**self._attrs, **t_attrs})
            index = index if (index is not None) else self._length
            if index > self._length:
                raise IndexError("index out of range")
            procs = []
            chunk_size = (len(sequence) + processes - 1) // processes
            pos = self._offset + index * self._element_size
            for ix in range(0, len(sequence), chunk_size):
                chunk = sequence[ix: ix + chunk_size]
                offset = pos + self._element_size * ix
                p = Process(target=self._write, args=(self._file, chunk, offset))
                procs.append(p)
                p.start()
            for p in procs:
                p.join()
            self._length = max(self._length, index + len(sequence))