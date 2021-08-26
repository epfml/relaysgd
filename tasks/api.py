import torch
from typing import Iterable, List, Mapping, Tuple

State = List[torch.Tensor]
Gradient = List[torch.Tensor]
Parameters = List[torch.Tensor]
Loss = float
Quality = Mapping[str, float]


class Batch:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __len__(self) -> int:
        return len(self._y)

    def to(self, device):
        return Batch(
            self._x.to(device)
            if type(self._x) is not dict
            else dict((_key, _value.to(device)) for _key, _value in self._x.items()),
            self._y.to(device),
        )


class Dataset:
    def random_split(self, fractions: List[float]) -> List["Dataset"]:
        pass

    def iterator(
        self, batch_size: int, shuffle: bool, repeat=True
    ) -> Iterable[Tuple[float, Batch]]:
        pass

    def __len__(self) -> int:
        pass


class Task:
    data: Dataset
    max_batch_size: int

    def initialize(self, seed: int) -> Tuple[Parameters, State]:
        pass

    def parameter_names(self) -> List[str]:
        pass

    def loss(
        self,
        parameters: List[torch.Tensor],
        state: State,
        batch: Batch,
        random_seed=None,
    ) -> Tuple[Loss, State]:
        pass

    def loss_and_gradient(
        self,
        parameters: List[torch.Tensor],
        state: State,
        batch: Batch,
        random_seed=None,
    ) -> Tuple[Loss, Gradient, State]:
        pass

    def quality(
        self,
        parameters: List[torch.Tensor],
        state: State,
        batch: Batch,
        random_seed=None,
    ) -> Quality:
        pass

    def evaluate(
        self, dataset: Dataset, parameters: List[torch.Tensor], state: State
    ) -> Quality:
        pass
