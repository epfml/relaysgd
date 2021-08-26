import contextlib
import itertools
import math
from typing import Iterable, List, NamedTuple, Tuple

import numpy as np
import torch

from utils.communication import get_rank, get_world_size

from .api import (Batch, Dataset, Gradient, Loss, Parameters, Quality, State,
                  Task)


class QuadraticsTask(Task):
    """avg_i ||A_i x - b_i||^2"""
    def __init__(self, d: float, non_iidness: float, sgd_noise_variance: float = 0.0):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = FakeDataset()
        self.max_batch_size = self.data.max_batch_size

        self.mean_num_data_per_worker = 1

        self._test_data = FakeDataset()

        self._zeta2 = non_iidness
        self._sgd_noise_variance = sgd_noise_variance
        self._d  = d

        num_workers = get_world_size()
        self.A = torch.stack(
            [
                1 / math.sqrt(num_workers) * torch.eye(d) * (i + 1)
                for i in range(0, num_workers)
            ]
        )
        with fork_rng_with_seed(42):
            self.B = torch.stack(
                [
                    torch.randn(d, 1) * math.sqrt(self._zeta2) / (i + 1)
                    for i in range(0, num_workers)
                ]
            )

        # Flatten all the parameters
        aa = self.A.view(-1, d)
        bb = self.B.view(-1, 1)
        self._target, _ = torch.solve(aa.T @ bb, aa.T @ aa)
        self._target = self._target.unsqueeze(0)

    def initialize(self, seed=42) -> Tuple[Parameters, State]:
        parameters = [torch.zeros_like(self._target, requires_grad=True)]
        state = []
        return parameters, state

    def loss(
        self,
        parameters: List[torch.Tensor],
        state: List[torch.Tensor],
        batch: Batch,
        random_seed=None,
    ) -> Tuple[Loss, State]:
        """L2 norm from optimum"""
        with torch.no_grad():
            return torch.sum((parameters[0] - self._target)**2)

    def loss_and_gradient(
        self,
        parameters: List[torch.Tensor],
        state: List[torch.Tensor],
        batch: Batch,
        random_seed=None,
    ) -> Tuple[Loss, Gradient, State]:
        loss = torch.sum((parameters[0] - self._target)**2)
        gradients = torch.autograd.grad(loss, parameters)

        # TODO: make sure the random seeds are not correlated between workers
        if self._sgd_noise_variance > 0:
            gradients[0].add_(torch.randn_like(gradients[0]), alpha=math.sqrt(self._sgd_noise_variance / self._d))
            
        return loss.item(), gradients, state

    def quality(
        self, parameters: List[torch.Tensor], state: List[torch.Tensor], batch: Batch
    ) -> Quality:
        return dict(distance_from_opt=self.loss(parameters, state, batch))

    def evaluate(
        self,
        dataset: Dataset,
        parameters: List[torch.Tensor],
        state: List[torch.Tensor],
    ) -> Quality:
        return self.quality(parameters, state, dataset)



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


@contextlib.contextmanager
def fork_rng_with_seed(seed):
    if seed is None:
        yield
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield
