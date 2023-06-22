import unittest
import tempfile
import torch

from dltool.hooks import *
from tests.fixtures import MockTrainer

SEED = 123456789
N = 5


class TestHooks(unittest.TestCase):
    def setUp(self):
        self.torch_gen = torch.random.manual_seed(SEED)
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir_path = Path(self.tmp_dir.name)
        self.models = [torch.nn.Linear(1,1) for _ in range(N)]
        self.m_values = torch.randn(N)

    def init(self):
        self.trainer = MockTrainer()
        self.trainer.logger.history = {"m": []}

    def test_watch_best_state_hook(self):
        min_hook = watch_best_state_hook("m", min)
        max_hook = watch_best_state_hook("m", max)

        def simulate_validations_steps(hooks):
            self.init()
            for m_value, model in zip(self.m_values, self.models):
                self.trainer.algorithm = model
                self.trainer.logger.history["m"].append(m_value)
                for hook in hooks:
                    hook(self.trainer)

        # min
        simulate_validations_steps([min_hook])
        self.assertEqual(self.trainer.best_model_state, self.models[0].state_dict())
        # max
        simulate_validations_steps([max_hook])
        self.assertEqual(self.trainer.best_model_state, self.models[3].state_dict())

    def test_save_state_hook(self):
        save_best_hook = save_state_hook(self.tmp_dir_path / "best.pt", best=True)
        save_hook = save_state_hook(self.tmp_dir_path / "current.pt", best=False)

        def simulate_validations_steps(hooks):
            self.init()
            for i, (m_value, model) in enumerate(zip(self.m_values, self.models)):
                self.trainer.algorithm = model
                self.trainer.best_model_state = self.models[-i-1].state_dict()
                self.trainer.logger.history["m"].append(m_value)
                for hook in hooks:
                    hook(self.trainer)
                self.assertEqual(self.trainer.best_model_state, torch.load(self.tmp_dir_path / "best.pt"))
                self.assertEqual(self.trainer.algorithm.state_dict(), torch.load(self.tmp_dir_path / "current.pt"))

        simulate_validations_steps([save_hook, save_best_hook])