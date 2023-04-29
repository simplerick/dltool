from collections.abc import Sequence, Callable


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


