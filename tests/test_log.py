import pytest
import numpy as np
import torch
from unittest.mock import MagicMock
from dltool.log import (
    History,
    Logger,
    format_dict_to_str,
)


def all_close(obj1, obj2, tol=1e-4):
    if isinstance(obj1, (int, float)):
        return abs(obj1 - obj2) < tol
    if isinstance(obj1, (list, tuple)):
        return all(all_close(o1, o2, tol) for o1, o2 in zip(obj1, obj2))
    if isinstance(obj1, dict):
        return all(all_close(obj1[k], obj2[k], tol) for k in obj1.keys() | obj2.keys())
    if isinstance(obj1, torch.Tensor):
        return torch.allclose(obj1, obj2, atol=tol)
    if isinstance(obj1, np.ndarray):
        return np.allclose(obj1, obj2, atol=tol)
    return obj1 == obj2


@pytest.fixture
def randn_tensor():
    tensor = torch.randn(3, 3)
    return tensor


@pytest.fixture
def randn_tensor_cuda():
    tensor = torch.randn(3, 3)
    try:
        tensor = tensor.cuda()
    except:
        pass
    return tensor


@pytest.fixture
def mock_backend():
    return MagicMock()


@pytest.fixture
def mock_console_logger():
    class MockConsoleLogger:
        def __init__(self):
            self.logged_messages = []

        def info(self, message):
            self.logged_messages.append(message)

    return MockConsoleLogger()


@pytest.fixture
def logger(mock_backend, mock_console_logger):
    return Logger(
        backend=mock_backend,
        console=True,
        in_memory=True,
        logging_freq=3
    )


def test_format_dict_to_str():
    d = {"a": 1.23}
    d_str = format_dict_to_str(d)
    assert "a" in d_str and "1.23" in d_str
    d = {"a": {"b": {"c": 1, "d": 2}, "e": 3}, "f": 4}
    d_str = format_dict_to_str(d)
    for c in "abcdef1234":
        assert c in d_str


def test_history():
    history = History()
    history[0].update({"a": 1, "b": 2})
    history[1].update({"a": 3, "c": {"a": 4}})
    history[2].update({"c": {"a": 5}})
    history[3].update({"c": {"b": 6}})
    assert all_close(history[1], {"a": 3, "c": {"a": 4}})
    assert all_close(history["b"], [2])
    assert all_close(history["a"], [1, 3])
    assert all_close(history["c"], [{"a": 4}, {"a": 5}, {"b": 6}])
    assert all_close(history["c/a"], [4, 5])


def test_logger_write_step(logger):
    logger.log({"loss": 0.1}, 10)
    logger.write(1)  # check that write can be performed with arbitrary step
    assert all_close(logger.history, {1: {"loss": 0.1}})
    logger.log({"loss": 0.2}, 20)
    logger.write(0)
    assert all_close(logger.history, {0: {"loss": 0.2}, 1: {"loss": 0.1}})


def test_logger_write_tensor(logger, randn_tensor_cuda):
    randn_tensor_cuda.requires_grad = True
    metrics = {"loss": 0.1, "loss_tensor": randn_tensor_cuda}
    logger.log(metrics, 0)
    logger.write(0)
    assert all_close(logger.history, {0: metrics})
    assert logger.history[0]["loss_tensor"].requires_grad == False
    assert logger.history[0]["loss_tensor"].device == torch.device("cpu")


def test_logger_log_names_conflict(logger):
    logger.log({"loss": 0}, group="test", step=0)
    logger.log({"accuracy": 1}, step=0)
    # invalid names
    with pytest.raises(ValueError):
        logger.log({"name\n": 1}, step=0)
    with pytest.raises(ValueError):
        logger.log({"group/metric": 1}, step=0)
    with pytest.raises(ValueError):
        logger.log({"metric": 1}, group="group\\", step=0)
    # conflicts
    with pytest.raises(ValueError):
        logger.log({"test": 1}, 0)
    with pytest.raises(ValueError):
        logger.log({"value": 1}, 0, group="accuracy")
    logger.log({"same": 1}, 0, group="same")
    logger.log({"other": 1}, 0, group="same")


def test_logger_log_auto_writing(logger, mock_backend):
    metrics = [{"loss": 0.5, "accuracy": 0.7}, {"loss": 0.4}, {"loss": 0.3, "accuracy": 0.9},
               {"accuracy": 1.0}]
    for step, m in enumerate(metrics):
        logger.log(m, step)
    expected_history = {2: {"loss": 0.4, "accuracy": 0.8}}
    assert len(mock_backend.log.call_args_list) == 1
    assert all_close(logger.history, expected_history)


def test_logger_not_persistent_metric_names(logger, randn_tensor):
    logger.log({"a": 4, "a_tensor": randn_tensor}, 0)  # writing step 2
    logger.log({"b": 5, "b_tensor": randn_tensor}, 3)  # writing step 5
    logger.log({"c": 6, "c_tensor": randn_tensor}, 6)
    expected_history = {2: {"a": 4, "a_tensor": randn_tensor},
                        5: {"b": 5, "b_tensor": randn_tensor}}
    assert all_close(logger.history, expected_history)


def test_logger_aggregations(logger, randn_tensor):
    logger.scalar_aggregation = max
    logger.tensor_aggregation = sum
    logger.log({"a": 4, "a_tensor": randn_tensor}, 0)
    logger.log({"a": 5, "a_tensor": randn_tensor}, 1)
    logger.write(0)
    expected_history = {0: {"a": 5, "a_tensor": 2 * randn_tensor}}
    assert all_close(logger.history, expected_history)


def test_logger_log_auto_writing_large_gaps(logger, mock_backend):
    logger.log({"loss": 0.3, "accuracy": 0.7}, 0)
    logger.log({"loss": 0.2, "accuracy": 0.8}, 30)
    logger.log({"loss": 0.1, "accuracy": 0.9}, 60)
    expected_history = {2: {"loss": 0.3, "accuracy": 0.7}, 32: {"loss": 0.2, "accuracy": 0.8}}
    assert len(mock_backend.log.call_args_list) == 2
    assert all_close(logger.history, expected_history)


def test_logger_log_forced_writing(logger, mock_backend):
    logger.log({"loss": 0.1}, 0)
    logger.log({"loss": 0.2}, 1, write=True)
    logger.log({"loss": 0.3}, 2)
    logger.log({"loss": 0.2}, 26, write=True)
    expected_history = {1: {"loss": 0.15}, 2: {"loss": 0.3}, 26: {"loss": 0.2}}
    assert len(mock_backend.log.call_args_list) == 3
    assert all_close(logger.history, expected_history)


def test_logger_log_accumulating(logger, mock_backend):
    logger.log({"loss": 0.1}, 0)
    logger.log({"loss": 0.2}, 1, accumulate=True)
    logger.log({"loss": 0.3}, 3, accumulate=True)
    assert len(mock_backend.log.call_args_list) == 0
    assert len(logger.history) == 0
    logger.log({"loss": 0.2}, 6)
    expected_history = {5: {"loss": 0.2}}
    assert len(mock_backend.log.call_args_list) == 1
    assert all_close(logger.history, expected_history)


def test_logger_log_chronology(logger):
    logger.log({"loss": 0.1}, 1, group="1", write=True)
    logger.log({"loss": 0.1}, 1, group="2")
    with pytest.raises(ValueError):
        logger.log({"loss": 0.1}, 1, group="1")
    logger.log({"accuracy": 0.9}, 1, group="1")


def test_logger_write_specific_groups_metrics(logger):
    logger.log({"loss": 0.1, "accuracy": 0.9}, 0, group="1")
    logger.log({"loss": 0.3, "accuracy": 0.7}, 1, group="2")
    logger.write(0, groups=["1"], metrics=["accuracy"])
    expected_history = {0: {"1": {"accuracy": 0.9}}}
    assert all_close(logger.history, expected_history)
    logger.write(1, groups=["2"], metrics=["loss"])
    expected_history |= {1: {"2": {"loss": 0.3}}}
    assert all_close(logger.history, expected_history)