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


def transformable(cls: type) -> type:
    """
    Modifies the given class in-place to enable data transformation.
    The transformed class will have a `with_transforms` method that attaches a sequence of transforms to the instance.
    The transforms will be applied to the data returned by the `__getitem__` method of the instance.
    """
    def _with_transforms(self, transforms: Sequence[Callable]):
        self.transforms = transforms
        return self

    def _getitem(self, *args, **kwargs):
        x = __getitem__(self, *args, **kwargs)
        if ts := getattr(self, "transforms", None):
            for t in ts:
                x = t(x)
        return x

    if hasattr(cls, "__getitem__") and not hasattr(cls, "with_transforms"):
        __getitem__ = cls.__getitem__
        cls.__getitem__ = _getitem
        cls.with_transforms = _with_transforms
    return cls

