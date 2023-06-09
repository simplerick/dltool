import json
import warnings
from copy import deepcopy
from io import DEFAULT_BUFFER_SIZE
from multiprocessing import Process
from pathlib import Path
from collections.abc import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import transformable

from torch.testing._internal.common_dtype import get_all_dtypes

DTYPE_SIZES = {
    str(dtype).split(".")[1]: torch.tensor(0, dtype=dtype).element_size()
    for dtype in get_all_dtypes(include_bfloat16=False)
}


@transformable
class FileSequenceDataset(Dataset, Sequence):
    _required_attrs = {'_dtype', '_shape'}
    __slots__ = ["_file", "_attrs", "_element_size", "_length", "_offset"] + list(_required_attrs)

    def __init__(self,
                 path: str | Path,
                 metadata: dict = None,
                 overwrite: bool = False,
                 chunk_size: int = DEFAULT_BUFFER_SIZE):
        self._file = Path(path)
        self._attrs = {}
        self._element_size = 0
        self._offset = 0
        self._length = 0
        self._sentinel = b'\n'

        if self._file.exists():
            if overwrite:
                warnings.warn(f"file {path} already exists and will be overwritten")
            else:
                with open(self._file, "rb") as f:
                    byte_str = []
                    while True:
                        chunk = f.read(chunk_size)
                        byte_str.append(chunk)
                        if not chunk or self._sentinel in chunk:
                            break
                    self._read_attrs(b"".join(byte_str))
        else:
            if metadata is None:
                raise FileNotFoundError(f"file {path} does not exist.")
        if metadata is not None:
            self.update_attrs(metadata)
            if metadata.keys() & FileSequenceDataset._required_attrs:
                warnings.warn(f"using {FileSequenceDataset._required_attrs} from the passed metadata. "
                              f"The file data may be decoded incorrectly.")

    @property
    def file(self):
        return str(self._file)

    @property
    def metadata(self):
        return deepcopy(self._attrs)

    def update_attrs(self, attrs):
        self._attrs.update(attrs)
        for key in FileSequenceDataset._required_attrs & self._attrs.keys():
            setattr(self, key, self._attrs[key])

    def _read_attrs(self, byte_str: bytes):
        try:
            sep_ix = byte_str.find(self._sentinel)
            if sep_ix == -1:
                raise RuntimeError
            self.update_attrs(json.loads(byte_str[:sep_ix]))
            if not self._attrs.keys() >= FileSequenceDataset._required_attrs:
                raise KeyError(f"metadata {self._attrs.keys()} doesn't contain required fields {FileSequenceDataset._required_attrs}")
            self._element_size = int(DTYPE_SIZES[self._dtype] * np.prod(self._shape))
            self._length = (self._file.stat().st_size - sep_ix - 1) // self._element_size
            self._offset = sep_ix + 1
        except Exception as e:
            warnings.warn(f"metadata for {self.file} is corrupted: \n{e}. If writing is attempted, the file will be overwritten!")

    def _write_metadata(self, metadata: dict):
        if self._file.exists():
            self._file.unlink()
        byte_str = json.dumps(metadata).encode("utf-8") + self._sentinel
        with open(self._file, "wb") as file:
            file.write(byte_str)
            file.flush()
        self._read_attrs(byte_str)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int | slice) -> torch.Tensor:
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
        t = t.squeeze(0) if isinstance(index, int) else t
        return t

    @staticmethod
    def _tensor_attrs(t: torch.Tensor) -> dict:
        dtype = str(t.dtype).split(".")[1]
        assert t.element_size() == DTYPE_SIZES[dtype]
        return {'_dtype': dtype,
                '_shape': list(t.size())}

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
