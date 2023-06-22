import torch.nn


class MockLogger:
    history: dict


class MockTrainer:
    def __init__(self):
        self.algorithm = torch.nn.Module()
        self.logger = MockLogger()
        self.best_model_state = dict()


