import warnings
import torch
import numpy as np

from dltool.algorithm import Algorithm
from dltool.data import DataIterator
from dltool.log import Logger
from dltool.utils import to, evaluating


class StopSignal(Exception):
    pass


class Trainer:
    def __init__(self, algorithm: Algorithm,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: object = None,
                 logger: Logger = None,
                 val_check_interval: int = 1.0
                 ):
        self.algorithm = algorithm
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger if logger is not None else Logger(None)
        self.val_check_interval = val_check_interval
        self.device = next(self.algorithm.parameters(), torch.empty(0)).device
        self._step_count = 0
        self.best_model_state = None
        self.hooks = []

    def opt_step(self, loss, optimizer, scheduler):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if self.scheduler is not None:
            scheduler.step()
            self.logger.log({"lr": self.scheduler.get_last_lr()[0]}, self._step_count, group="hparams")

    def loop(self, num_steps, dataloader, step_fn):
        for batch_idx in range(num_steps):
            batch = to(next(dataloader), device=self.device)
            loss, metrics = step_fn(batch, self._step_count)
            # opt step
            if loss is not None and loss.requires_grad:
                self.opt_step(loss, self.optimizer, self.scheduler)
            # log metrics
            metrics = to(metrics, device='cpu')
            fn_name = step_fn.__name__
            self.logger.log(metrics, self._step_count, group=fn_name,
                            write=(fn_name in ['val_step', 'test_step'] and batch_idx == num_steps - 1))
            if fn_name == 'train_step':
                self._step_count += 1
                self.logger.log({"Epoch": self._step_count / len(dataloader)}, self._step_count)

    def fit(self, epochs, train_dataloader, val_dataloader=None):
        # data iterators
        train_iterator = DataIterator(train_dataloader)
        val_iterator = DataIterator(val_dataloader) if val_dataloader is not None else None
        # check train dataloader
        if len(train_dataloader) > 1 and not train_dataloader.drop_last:
            warnings.warn("the last incomplete batch in train dataloader is not dropped.")
        self._step_count = 0
        # split training steps
        train_chunks = np.array_split(np.arange(epochs * len(train_dataloader)),
                                      max(1, int(epochs / self.val_check_interval)))
        for chunk in train_chunks:
            if len(chunk) == 0:
                continue
            self.loop(len(chunk), train_iterator, self.algorithm.train_step)
            if val_dataloader is not None:
                with evaluating(self.algorithm):
                    self.loop(len(val_dataloader), val_iterator, self.algorithm.val_step)
            # hooks
            try:
                for fn in self.hooks:
                    fn(self)
            except StopSignal as e:  # allows to stop training from hook
                self.logger.console.info(f"{e.__class__.__name__}: {e}" if e.args else f"{e.__class__.__name__}")
                break
        # load best model if any is saved
        if self.best_model_state is not None:
            self.algorithm.load_state_dict(self.best_model_state)

    def test(self, test_dataloader):
        # data iterator
        test_iterator = DataIterator(test_dataloader)
        # testing loop
        with evaluating(self.algorithm):
            self.loop(len(test_dataloader), test_iterator, self.algorithm.test_step)

    # def sanity_check(self, batch, max_iter=100, criterion=1e-4):
    #     dl = torch.utils.data.DataLoader([batch])
    #     self.fit(epochs=max_iter, train_dataloader=dl)
    #     return {k: v[-10:] for k, v in self.history.items()}
