import itertools
from typing import Iterable, List, NamedTuple, Tuple

import torch

from .api import (Batch, Dataset, Gradient, Loss, Parameters, Quality, State,
                  Task)

"""
This task is meant to verify correctness of algorithm implementations.
By studying how a gradient from one worker propagates to the rest, we can compare it to our toy implementations.
"""

class DeliveryTask(Task):
    def __init__(self, source_worker=None):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = FakeDataset()
        self.max_batch_size = self.data.max_batch_size

        self.mean_num_data_per_worker = 1

        self._test_data = FakeDataset()

        if source_worker:
            self._source_worker = source_worker
        else:
            self._source_worker = torch.distributed.get_world_size() // 3

    def initialize(self, seed=42) -> Tuple[Parameters, State]:
        parameters = [torch.zeros(size=[], device=self._device)]
        state = []
        return parameters, state

    def loss(
        self,
        parameters: List[torch.Tensor],
        state: List[torch.Tensor],
        batch: Batch,
        random_seed=None,
    ) -> Tuple[Loss, State]:
        return torch.zeros(size=[], device=self._device), state

    def loss_and_gradient(
        self,
        parameters: List[torch.Tensor],
        state: List[torch.Tensor],
        batch: Batch,
        random_seed=None,
    ) -> Tuple[Loss, Gradient, State]:
        if batch.idx == 0 and torch.distributed.get_rank() == self._source_worker:
            grad = [-torch.ones_like(parameters[0]) * torch.distributed.get_world_size()]
        else:
            grad = [torch.zeros_like(parameters[0])]

        return 0.0, grad, state

    def quality(
        self, parameters: List[torch.Tensor], state: List[torch.Tensor], batch: Batch
    ) -> Quality:
        return dict()

    def evaluate(
        self,
        dataset: Dataset,
        parameters: List[torch.Tensor],
        state: List[torch.Tensor],
    ) -> Quality:
        return dict(param=parameters[0].item())

class Batch(NamedTuple):
    idx: int

class FakeDataset:
    max_batch_size = 1
    def __len__(self):
        return 1

    def iterator(
        self, batch_size: int, shuffle=True, repeat=True, ref_num_data=None
    ) -> Iterable[Tuple[float, Batch]]:
        for i in itertools.count() if repeat else [0]:
            yield i, Batch(idx=i)
