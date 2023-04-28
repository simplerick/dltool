from collections.abc import Sequence
from itertools import accumulate
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd



@runtime_checkable
class TimeSequence(Protocol):
    start_time: pd.Timestamp
    end_time:   pd.Timestamp
    time_delta: pd.Timedelta
    def __len__(self) -> int: pass
    def __getitem__(self, item: int | slice): pass



class TimeWindowSequence:
    def __init__(self,
                 sequence:   TimeSequence,
                 intervals:  pd.Timedelta | str | Sequence[pd.Timedelta | str],
                 start:      pd.Timestamp | str = None,
                 end:        pd.Timestamp | str = None,
                 stride:     pd.Timedelta = None):
        self.sequence = sequence
        self.start = pd.Timestamp(start) if start else sequence.start_time
        self.end = pd.Timestamp(end) if end else sequence.end_time
        self.stride = pd.Timedelta(stride) if stride else sequence.time_delta
        self.date_range = pd.date_range(sequence.start_time, periods=len(sequence), freq=sequence.time_delta)

        self.reduce = False
        if isinstance(intervals, (pd.Timedelta, str)):
            intervals = [intervals]
            self.reduce = True
        self.intervals = [pd.Timedelta(x) for x in intervals]
        if not all([interval.value % sequence.time_delta.value == 0 for interval in self.intervals]):
            raise ValueError("Interval durations must be a multiple of a sequence time delta")

        self.length = (self.end - self.start - np.sum(self.intervals)) // self.stride + 1

        self._t_start = self.start.value - sequence.start_time.value
        self._t_delta = sequence.time_delta.value
        self._t_stride = self.stride.value
        self._t_edges = list(accumulate([0, *[interval.value for interval in self.intervals]]))

    def __len__(self) -> int:
        return self.length

    def _get_index_positions(self, index: int) -> list[int]:
        if index >= len(self):
            raise IndexError("out of range")
        pos = [(self._t_start + index * self._t_stride + t_i + self._t_delta - 1) // self._t_delta
               for t_i in self._t_edges]
        return pos

    def __getitem__(self, index: int) -> list[Sequence] | Sequence:
        if isinstance(index, slice):
            return NotImplemented
        pos = self._get_index_positions(index)
        seqs = [self.sequence[pos[i]:pos[i + 1]] for i in range(len(pos) - 1)]
        if self.reduce:
            seqs, _ = seqs
        return seqs

    def get_dates(self, index: int) -> list[pd.DatetimeIndex] | pd.DatetimeIndex:
        pos = self._get_index_positions(index)
        seqs = [self.date_range[pos[i]:pos[i + 1]] for i in range(len(pos) - 1)]
        if self.reduce:
            seqs, _ = seqs
        return seqs



class TimeWindowsSequence:
    def __init__(self,
                 sequences:  TimeSequence | Sequence[TimeSequence],
                 intervals:  pd.Timedelta | str | Sequence[pd.Timedelta | str],
                 stride:     pd.Timedelta = None):
        self.reduce_s = False
        if isinstance(sequences, TimeSequence):
            sequences = [sequences]
            self.reduce_s = True
        self.sequences = sequences
        self.reduce_i = False
        if isinstance(intervals, (pd.Timedelta, str)):
            intervals = [intervals]
            self.reduce_i = True
        self.intervals = [pd.Timedelta(x) for x in intervals]

        self.start = max(s.start_time for s in sequences)
        self.end = min(s.end_time for s in sequences)
        self.stride = pd.Timedelta(stride) if stride else max([s.time_delta for s in sequences])
        self.date_range = pd.date_range(self.start, self.end, freq=self.stride, inclusive="left")
        self.length = (self.end - self.start - np.sum(self.intervals)) // self.stride + 1
        self.slice_seq = [TimeWindowSequence(x, intervals, self.start, stride=stride) for x in sequences]

    def __len__(self) -> int:
        return self.length

    def _reduce_dimensions(self, slices):
        if self.reduce_s:
            slices = [s[0] for s in slices]
        if self.reduce_i:
            slices = slices[0]
        return slices

    def __getitem__(self, index: int) -> list[Sequence] | Sequence:
        if isinstance(index, slice):
            return NotImplemented
        slices = [s[index] for s in self.slice_seq]  # [sequences, intervals]
        # transpose
        seqs = [[s[i] for s in slices] for i in range(len(self.intervals))]  # [intervals, sequences]
        # reduce dimensions
        seqs = self._reduce_dimensions(seqs)
        return seqs

    def get_dates(self, index: int) -> list[pd.DatetimeIndex] | pd.DatetimeIndex:
        slices = [s.get_dates(index) for s in self.slice_seq]  # [sequences, intervals]
        # transpose
        seqs = [[s[i] for s in slices] for i in range(len(self.intervals))]  # [intervals, sequences]
        # reduce dimensions
        seqs = self._reduce_dimensions(seqs)
        return seqs

    def get_indices_from_timerange(self, start, end):
        """
        Get all indices for which the sampled time period fall inside the designated date segment
        """
        # rightmost position
        start = pd.Timestamp(start)
        end_date = pd.Timestamp(end) - np.sum(self.intervals)
        return range(min(self.length, np.searchsorted(self.date_range, start)),
                     min(self.length, np.searchsorted(self.date_range, end_date, 'right')))

    def get_indices_from_dates(self, dates: Sequence[pd.Timestamp]):
        """
        Get indices of target dates in datetime index
        """
        indices = self.date_range.get_indexer_for(dates)
        indices = np.where(indices < self.length, indices, np.nan).astype(np.int32)
        return indices
