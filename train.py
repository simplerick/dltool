import warnings
import torch
from dltool.data import get_batch
from dltool.log import Logger
from dltool.utils import to, detach, evaluating, cpu_state_dict
import numpy as np


class Trainer:
    def __init__(self, algorithm, optimizer, scheduler, logger, val_check_interval=1.0, log_interval=10):
        self.algorithm = algorithm
        self.model = algorithm.model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.val_check_interval = val_check_interval
        self.device = next(self.model.parameters(), torch.empty(0)).device
        self.logger = Logger(logger, log_interval)
        self.logger.api.define_metric("Epoch", hidden=True)
        self._step_count = 0
        self.best_model_state = None
        self.val_hooks = []

    def fix_best_state(self):
        self.best_model_state = cpu_state_dict(self.model)

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
            metrics = to(metrics, device='cpu')
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
            warnings.warn("the last incomplete batch in train dataloader is not dropped.")
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
            # hooks
            try:
                for fn in self.val_hooks:
                    fn(self)
            except Exception as e:
                warnings.warn(str(e))
                break
        # load best model if any is saved
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    def test(self, test_dataloader):
        # testing loop
        with evaluating(self.model):
            self.loop(len(test_dataloader), test_dataloader, self.algorithm.test_step)

    def set_opt_goal(self, metric, select_fn=min):
        def save_model_checkpoints(obj):
            if len(obj.logger.history[metric]) > 0 and select_fn(self.logger.history[metric]) == self.logger.history[metric][-1]:
                obj.fix_best_state()
        self.val_hooks.append(save_model_checkpoints)


    # def sanity_check(self, batch, max_iter=100, criterion=1e-4):
    #     dl = torch.utils.data.DataLoader([batch])
    #     self.fit(epochs=max_iter, train_dataloader=dl)
    #     return {k: v[-10:] for k, v in self.history.items()}
