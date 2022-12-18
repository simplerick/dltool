import string
import tempfile
import unittest
from pathlib import Path
from random import Random

import torch

from dltool.data import SequenceDataset





def generate_random_string(length, random):
    return ''.join(random.choice(string.ascii_uppercase) for _ in range(length))


class TestSequenceDataset(unittest.TestCase):

    def setUp(self):
        self.random = Random(0)
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir_path = Path(self.tmp_dir.name)

    def _check_data(self, dataset, data, metadata = None):
        if metadata is not None:
            self.assertTrue(metadata.items() <= dataset.metadata.items())
        self.assertEqual(len(dataset), len(data))
        self.assertTrue((dataset[:] == data).all())

    def _new_metadata(self):
        return {generate_random_string(10, self.random): generate_random_string(10, self.random)
                for _ in range(10)}

    def _new_data(self):
        return torch.randn(10, 2, 2)

    def test_write_load(self):
        """
        Creates a SequenceDataset with a random metadata in a tmp folder, writes a torch tensor
        and checks that it can be loaded, metadata is correct and data is correct.
        """
        path = self.tmp_dir_path / "test_create_load.seq"
        data = self._new_data()
        metadata = self._new_metadata()
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
        dataset = SequenceDataset(path, self._new_metadata())
        dataset.write(self._new_data())
        new_metadata = self._new_metadata()
        new_data = self._new_data()
        with self.assertRaises(RuntimeError):
            SequenceDataset(path, new_metadata, overwrite=False)
        dataset = SequenceDataset(path, new_metadata, overwrite=True)
        dataset.write(new_data)
        self._check_data(dataset, new_data, new_metadata)
        dataset = SequenceDataset(path)
        with self.assertRaises(RuntimeError):
            SequenceDataset(path.parent / "non_existing_file.seq")






if __name__ == "__main__":
  unittest.main()




