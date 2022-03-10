import warnings
from tqdm import tqdm
import torch
from dltool.data import get_batch
from dltool.log import Logger
from dltool.utils import to, detach, evaluating
from collections import defaultdict
import numpy as np


class Trainer:
    def __init__(self, algorithm, optimizer, scheduler, logger, val_check_interval=1.0, log_interval=10):
        self.algorithm = algorithm
        self.model = algorithm.model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.val_check_interval = val_check_interval
        self.device = next(self.model.parameters(), torch.empty(0)).device
        self.best_model_state = to(self.model.state_dict(), device='cpu')
        self._step_count = 0
        self.logger = Logger(logger, log_interval)
        self.logger.api.define_metric("Epoch", hidden=True)

    def opt_step(self, loss, optimizer, scheduler):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    def loop(self, num_steps, dataloader, step_fn, optimize=False):
        for batch_idx in range(num_steps):
            batch = to(get_batch(dataloader), device=self.device)
            loss, metrics = step_fn(batch, self._step_count)
            metrics = detach(metrics)  # detach tensors if they are attached to graph
            fn_name = step_fn.__name__
            self.logger.log(metrics, self._step_count, group=fn_name)
            if optimize:
                self.opt_step(loss, self.optimizer, self.scheduler)
                self._step_count += 1
                self.logger.log({"Epoch": self._step_count / len(dataloader)}, self._step_count)
                self.logger.log({"lr": self.scheduler.get_last_lr()[0]}, self._step_count, group="hparams")

    def fit(self, epochs, train_dataloader, val_dataloader=None):
        # check train dataloader
        if len(train_dataloader) > 1 and not train_dataloader.drop_last:
            warnings.warn("The last incomplete batch in train dataloader is not dropped.")
        self._step_count = 0
        # split training steps
        train_chunks = np.array_split(np.arange(epochs * len(train_dataloader)),
                                      max(1, int(epochs / self.val_check_interval)))
        for chunk in train_chunks:
            self.loop(len(chunk), train_dataloader, self.algorithm.train_step, optimize=True)
            if val_dataloader is not None:
                print("Start validation", self._step_count)
                with evaluating(self.model):
                    self.loop(len(val_dataloader), val_dataloader, self.algorithm.val_step)
                self.logger.flush()

    def test(self, test_dataloader):
        # testing loop
        with evaluating(self.model):
            self.loop(len(test_dataloader), test_dataloader, self.algorithm.test_step)

    # def sanity_check(self, batch, max_iter=100, criterion=1e-4):
    #     dl = torch.utils.data.DataLoader([batch])
    #     self.fit(epochs=max_iter, train_dataloader=dl)
    #     return {k: v[-10:] for k, v in self.history.items()}
