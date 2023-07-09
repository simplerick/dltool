import string
import tempfile
import unittest
from pathlib import Path
from random import Random

import torch
import pandas as pd

from dltool.data import FileSequenceDataset, transformable, TimeWindowsSequence

SEED = 123456789

def generate_random_string(length, random):
    return ''.join(random.choice(string.ascii_uppercase) for _ in range(length))


def _new_metadata(random, length=10):
    return {generate_random_string(length, random): generate_random_string(length, random)
            for _ in range(10)}


def _new_data(random, length=10):
    return torch.randn(length, 2, 2, generator=random)



class TestFileSequenceDataset(unittest.TestCase):
    def setUp(self):
        self.random = Random(SEED)
        self.torch_gen = torch.random.manual_seed(SEED)
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir_path = Path(self.tmp_dir.name)

    def _check_metadata(self, m1, m2):
        self.assertEqual(
            {k: v for k, v in m1.items() if k not in FileSequenceDataset._required_attrs}.items(),
            {k: v for k, v in m2.items() if k not in FileSequenceDataset._required_attrs}.items()
        )

    def _check_data(self, dataset, data, metadata = None):
        if metadata is not None:
            self._check_metadata(metadata, dataset.metadata)
        self.assertEqual(len(dataset), len(data))
        self.assertTrue((dataset[:] == data).all())

    def test_write_load(self):
        """
        Creates a SequenceDataset with a random metadata in a tmp folder, writes a torch tensor
        and checks that it can be loaded, metadata is correct and data is correct.
        """
        path = self.tmp_dir_path / "test_create_load.seq"
        data = _new_data(self.torch_gen)
        metadata = _new_metadata(self.random)
        # check creating, writing, and loading works
        dataset = FileSequenceDataset(path, metadata)
        dataset.write(data, processes=2)
        self._check_data(dataset, data, metadata)
        dataset2 = FileSequenceDataset(path)
        self._check_data(dataset2, data, dataset.metadata)
        # try to write to a specific index
        dataset.write(data[6:10], index=3, processes=2)
        dataset.write(data[3:6], index=7, processes=2)
        self._check_data(dataset, torch.cat([data[:3], data[6:10], data[3:6]]))

    def test_overwrite_and_loading(self):
        """
        Checks that if file is exists, the file will be overwritten or not depending on the `overwrite` flag.
        Checks that if metadata is not provided and the file does not exist, the error is raised.
        """
        path = self.tmp_dir_path / "test_overwrite_and_loading.seq"
        metadata = _new_metadata(self.random)
        data = _new_data(self.torch_gen)
        dataset = FileSequenceDataset(path, metadata)
        dataset.write(data)
        new_metadata = _new_metadata(self.random)
        new_data = _new_data(self.torch_gen)

        dataset = FileSequenceDataset(path, new_metadata)
        self._check_data(dataset, data, metadata | new_metadata)

        for attr in FileSequenceDataset._required_attrs:
            with self.assertWarns(Warning):
                dataset = FileSequenceDataset(path, {attr: None}, overwrite=False)
                self.assertTrue(dataset._attrs[attr] is None)

        dataset = FileSequenceDataset(path, new_metadata, overwrite=True)
        dataset.write(new_data)
        self._check_data(dataset, new_data, new_metadata)
        FileSequenceDataset(path)

        with self.assertRaises(FileNotFoundError):
            FileSequenceDataset(path.parent / "non_existing_file.seq")



class TestTransformable(unittest.TestCase):
    def setUp(self):
        self.random = Random(SEED)
        self.torch_gen = torch.random.manual_seed(SEED)

        @transformable
        class TensorDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data

            def __getitem__(self, index):
                return self.data[index]

        @transformable
        class A:
            pass

        self.transformable_class = TensorDataset
        self.no_getitem_class = A

    def test_transform(self):
        """Checks that the transforms are applied to the data"""
        data = _new_data(self.torch_gen, 2)
        transform1 = lambda x: x + 1
        transform2 = lambda x: 2 * x
        ds = self.transformable_class(data)
        self.assertTrue(not hasattr(ds, "transforms"))
        ds.with_transforms([transform1, transform2])
        self.assertTrue(ds.transforms == [transform1, transform2])
        self.assertTrue((ds[:] == transform2(transform1(data))).all())
        ds = self.no_getitem_class()
        self.assertTrue(not hasattr(ds, "with_transforms"))

    def test_instance_sharing(self):
        """Check that the transforms are not shared between instances"""
        transform1 = lambda x: x + 1
        transform2 = lambda x: 2 * x
        ds1 = self.transformable_class(torch.empty(1)).with_transforms([transform1])
        ds2 = self.transformable_class(torch.empty(1)).with_transforms([transform2])
        self.assertTrue(ds1.transforms == [transform1])
        self.assertTrue(ds2.transforms == [transform2])



class TestTimeWindowsSequence(unittest.TestCase):
    def setUp(self):
        class TimeSeq(pd.DatetimeIndex): pass

        start_time = pd.Timestamp("2022-01-01 00:00:00")
        time_delta = pd.Timedelta(1, "h")
        ts = TimeSeq([start_time + i * time_delta for i in range(10)])
        ts.start_time, ts.time_delta = start_time, time_delta
        ts.end_time = ts.start_time + len(ts) * ts.time_delta  # 2022-01-01 10:00:00

        small_delta = pd.Timedelta(1, 's')
        start_time2 = ts.start_time + small_delta  # 2022-01-01 00:00:01
        time_delta2 = 2 * ts.time_delta
        ts2 = TimeSeq([start_time2 + i * time_delta2 for i in range(10)])
        ts2.start_time, ts2.time_delta = start_time2, time_delta2
        ts2.end_time = ts2.start_time + len(ts2) * ts2.time_delta  # 2022-01-01 20:00:01

        stride = pd.Timedelta(30, 'm')
        intervals = [ts2.time_delta, 2 * ts2.time_delta] # 2h, 4h
        self.time_window_seq = TimeWindowsSequence([ts, ts2], intervals, stride=stride)


    def test_date_range(self):
        self.assertEqual(self.time_window_seq.start, pd.Timestamp("2022-01-01 00:00:01"))
        self.assertEqual(self.time_window_seq.end, pd.Timestamp("2022-01-01 10:00:00"))
        self.assertSequenceEqual(self.time_window_seq.intervals, [pd.Timedelta(2, 'h'), pd.Timedelta(4, 'h')])
        self.assertEqual(self.time_window_seq.reduce_s, False)
        self.assertEqual(self.time_window_seq.reduce_i, False)
        self.assertTrue(
            (self.time_window_seq.date_range == pd.date_range(self.time_window_seq.start, self.time_window_seq.end,
                                                              freq="30min", inclusive='left')).all())

    def test_len(self):
        self.assertEqual(len(self.time_window_seq), 8)

    def test_getitem_get_dates(self):
        def check(x, y):
            [x00, x01], [x10, x11] = x
            [y00, y01], [y10, y11] = y
            for a,b in zip([x00, x01, x10, x11], [y00, y01, y10, y11]):
                self.assertTrue((a == b).all())

        el0 = [
            [pd.DatetimeIndex(['2022-01-01 01:00:00', '2022-01-01 02:00:00']),
             pd.DatetimeIndex(['2022-01-01 00:00:01'])],
            [pd.DatetimeIndex(['2022-01-01 03:00:00', '2022-01-01 04:00:00', '2022-01-01 05:00:00', '2022-01-01 06:00:00']),
             pd.DatetimeIndex(['2022-01-01 02:00:01', '2022-01-01 04:00:01'])]
        ]
        check(self.time_window_seq[0], el0)
        check(self.time_window_seq.get_dates(0), el0)

        el3 = [
            [pd.DatetimeIndex(['2022-01-01 02:00:00', '2022-01-01 03:00:00']),
             pd.DatetimeIndex(['2022-01-01 02:00:01'])],
            [pd.DatetimeIndex(['2022-01-01 04:00:00', '2022-01-01 05:00:00', '2022-01-01 06:00:00', '2022-01-01 07:00:00']),
             pd.DatetimeIndex(['2022-01-01 04:00:01', '2022-01-01 06:00:01'])]
        ]
        check(self.time_window_seq[3], el3)
        check(self.time_window_seq.get_dates(3), el3)

        for i in range(len(self.time_window_seq)):
            left_bound = self.time_window_seq.date_range[i]
            right_bound = left_bound + sum(self.time_window_seq.intervals, pd.Timedelta(0))
            el = self.time_window_seq.get_dates(i)
            for x in el:
                for y in x:
                    self.assertTrue(left_bound <= y[0] and y[-1] < right_bound)

    def test_get_indices_from_timerange_and_dates(self):
        left_bound = pd.Timestamp("2022-01-01 03:30:00")
        right_bound = pd.Timestamp("2022-01-01 08:40:00")
        indices_timerange = self.time_window_seq.get_indices_from_timerange(left_bound, right_bound)
        indices_dates = self.time_window_seq.get_indices_from_dates([left_bound, right_bound])
        self.assertEqual(indices_timerange, range(*indices_dates))
        for i in indices_timerange:
            el = self.time_window_seq.get_dates(i)
            for x in el:
                for y in x:
                    self.assertTrue(left_bound <= y[0] and y[-1] < right_bound)


if __name__ == "__main__":
  unittest.main()




