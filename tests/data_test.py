import string
import tempfile
import unittest
from pathlib import Path
from random import Random

import torch

from dltool.data import SequenceDataset, transformable

SEED = 123456789

def generate_random_string(length, random):
    return ''.join(random.choice(string.ascii_uppercase) for _ in range(length))


def _new_metadata(random, length=10):
    return {generate_random_string(length, random): generate_random_string(length, random)
            for _ in range(10)}


def _new_data(random, length=10):
    return torch.randn(length, 2, 2, generator=random)



class TestSequenceDataset(unittest.TestCase):
    def setUp(self):
        self.random = Random(SEED)
        self.torch_gen = torch.random.manual_seed(SEED)
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir_path = Path(self.tmp_dir.name)

    def _check_data(self, dataset, data, metadata = None):
        if metadata is not None:
            self.assertTrue(metadata.items() <= dataset.metadata.items())
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
        dataset = SequenceDataset(path, metadata)
        dataset.write(data, processes=2)
        self._check_data(dataset, data, metadata)
        dataset2 = SequenceDataset(path)
        self._check_data(dataset2, data, dataset.metadata)
        # try to write to a specific index
        dataset.write(data[6:10], index=3, processes=2)
        dataset.write(data[3:6], index=7, processes=2)
        self._check_data(dataset, torch.cat([data[:3], data[6:10], data[3:6]]))

    def test_overwrite_and_loading(self):
        """
        Checks that if metadata is provided, the file will be overwritten or not depending on the `overwrite` flag.
        Checks that if metadata is not provided and the file does not exist, the error is raised.
        """
        path = self.tmp_dir_path / "test_overwrite_and_loading.seq"
        dataset = SequenceDataset(path, _new_metadata(self.random))
        dataset.write(_new_data(self.torch_gen))
        new_metadata = _new_metadata(self.random)
        new_data = _new_data(self.torch_gen)
        with self.assertRaises(RuntimeError):
            SequenceDataset(path, new_metadata, overwrite=False)
        dataset = SequenceDataset(path, new_metadata, overwrite=True)
        dataset.write(new_data)
        self._check_data(dataset, new_data, new_metadata)
        SequenceDataset(path)
        with self.assertRaises(RuntimeError):
            SequenceDataset(path.parent / "non_existing_file.seq")



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

        self.transformable_class = TensorDataset

    def test_transform(self):
        """Checks that the transforms are applied to the data"""
        data = _new_data(self.torch_gen, 2)
        transform1 = lambda x: x + 1
        transform2 = lambda x: 2 * x
        ds = self.transformable_class(data)
        self.assertTrue(not hasattr(ds, "transforms"))
        ds = ds.with_transforms([transform1, transform2])
        self.assertTrue(ds.transforms == [transform1, transform2])
        self.assertTrue((ds[:] == transform2(transform1(data))).all())

    def test_instance_sharing(self):
        """Check that the transforms are not shared between instances"""
        transform1 = lambda x: x + 1
        transform2 = lambda x: 2 * x
        ds1 = self.transformable_class(torch.empty(1)).with_transforms([transform1])
        ds2 = self.transformable_class(torch.empty(1)).with_transforms([transform2])
        self.assertTrue(ds1.transforms == [transform1])
        self.assertTrue(ds2.transforms == [transform2])







if __name__ == "__main__":
  unittest.main()




