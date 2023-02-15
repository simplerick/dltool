from abc import ABC, abstractmethod
import torch
import warnings
import re


class Algorithm(ABC, torch.nn.Module):
    """
    Inherit this class to define custom dl algorithm.
    """
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_trainable_modules(self, names: [str]) -> list:
        """
        Enable gradients for the modules, whose names match one of the given patterns, and disable gradients for the rest.
        Applies recursively to all children of the matched modules.

        Args:
            names: list of patterns to match module names

        Returns:
            names of the modules, whose gradients were enabled
        """
        self.requires_grad_(False)
        self.eval()
        training_modules = []
        p_templates = [re.compile(f"{n}\.|{n}$") for n in names]

        def traverse(name, mod):
            for pattern in p_templates:
                if re.match(pattern, name):
                    training_modules.append(name)
                    mod.requires_grad_(True)
                    mod.train()
                    return
            for n, m in mod.named_children():
                traverse(f"{name}.{n}", m)

        for n, m in self.named_children():
            traverse(n, m)
        return training_modules

    @abstractmethod
    def train_step(self, batch, step_idx: int) -> (torch.Tensor, dict):
        pass

    def val_step(self, batch, step_idx: int) -> (torch.Tensor, dict):
        warnings.warn("the default behavior of the `val_step` is the same as `train_step`")
        return self.train_step(batch, step_idx)

    def test_step(self, batch, step_idx: int) -> (torch.Tensor, dict):
        warnings.warn("the default behavior of the `test_step` is the same as `val_step`")
        return self.val_step(batch, step_idx)
