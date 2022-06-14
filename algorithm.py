from abc import ABC, abstractmethod
import torch
import warnings
import re


class Algorithm(ABC, torch.nn.Module):
    """
    Inherit this class to define custom dl algorithm.
    """
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        for k, v in kwargs.items():
            setattr(self, k, v)

    # def set_training_parts(self, parameters: [str]) -> list:
    #     self.model.requires_grad_(False)
    #     training_parameters = []
    #     for p_template in parameters:
    #         pattern = re.compile(f"{p_template}\.|{p_template}$")
    #         for n, m in self.model.named_parameters():
    #             if re.match(pattern, n):
    #                 training_parameters.append(n)
    #                 m.requires_grad_(True)
    #     return training_parameters

    @abstractmethod
    def train_step(self, batch, step_idx: int) -> (torch.Tensor, dict):
        pass

    def val_step(self, batch, step_idx: int) -> (torch.Tensor, dict):
        warnings.warn("the default behavior of the `val_step` is the same as `train_step`")
        return self.train_step(batch, step_idx)

    def test_step(self, batch, step_idx: int) -> (torch.Tensor, dict):
        warnings.warn("the default behavior of the `test_step` is the same as `val_step`")
        return self.val_step(batch, step_idx)
