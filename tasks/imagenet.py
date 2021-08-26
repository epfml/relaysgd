import os
import contextlib
import itertools
import re
from typing import Iterable, List, Tuple

import torch
from torch.random import fork_rng
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset, random_split
import torch
import torchvision.transforms as transforms

from .api import Batch, Dataset, Gradient, Loss, Parameters, Quality, State, Task
from .cifar import PyTorchDataset, parameter_type, fork_rng_with_seed
from .utils.preprocess_toolkit import get_transform
from .utils.lmdb import LMDBPT


class ImageNetTask(Task):
    def __init__(
        self, weight_decay, model_name, data_split_method, non_iid_alpha=None, seed=0
    ):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = ImageNetDataset("train", device=self._device)
        self.max_batch_size = self.data.max_batch_size

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Splitting data by worker
            num_workers = torch.distributed.get_world_size()
            if data_split_method == "dirichlet":
                splits = self.data.dirichlet_split(
                    num_workers, non_iid_alpha, seed=seed
                )
            elif data_split_method == "random":
                splits = self.data.random_split(
                    fractions=[1 / num_workers for _ in range(num_workers)], seed=seed
                )
            else:
                raise ValueError(
                    f"Unknown value {data_split_method} for data_split_method"
                )
            self.mean_num_data_per_worker = (
                sum(len(split) for split in splits) / num_workers
            )
            print(
                f"Splitting data using {data_split_method} according to",
                [len(split) for split in splits],
            )
            self.data = splits[torch.distributed.get_rank()]
        else:
            self.mean_num_data_per_worker = len(self.data)

        self._test_data = ImageNetDataset("test", device=self._device)

        self._model_name = model_name
        self._model = self._create_model()
        self._criterion = torch.nn.CrossEntropyLoss().to(self._device)

        self._weight_decay_per_param = [
            weight_decay for p, _ in self._model.named_parameters()
        ]

    def initialize(self, seed=42) -> Tuple[Parameters, State]:
        with fork_rng_with_seed(seed):
            self._model = self._create_model()
        parameters = [p.data for p in self._model.parameters()]
        state = [b.data for b in self._model.buffers()]
        return parameters, state

    def loss(
        self,
        parameters: List[torch.Tensor],
        state: List[torch.Tensor],
        batch: Batch,
        random_seed=None,
    ) -> Tuple[Loss, State]:
        with torch.no_grad():
            with fork_rng_with_seed(random_seed):
                output, state = self.forward(
                    batch._y, parameters, state, is_training=True
                )
        loss = self._criterion(output, batch._y).item()
        return loss, state

    def loss_and_gradient(
        self,
        parameters: List[torch.Tensor],
        state: List[torch.Tensor],
        batch: Batch,
        random_seed=None,
    ) -> Tuple[Loss, Gradient, State]:
        with fork_rng_with_seed(random_seed):
            output, state = self._forward(batch._x, parameters, state, is_training=True)
        loss = self._criterion(output, batch._y)
        gradients = torch.autograd.grad(loss, list(self._model.parameters()))

        for g, wd, p in zip(gradients, self._weight_decay_per_param, parameters):
            g.add_(p, alpha=wd)

        return loss.item(), gradients, state

    def quality(
        self, parameters: List[torch.Tensor], state: List[torch.Tensor], batch: Batch
    ) -> Quality:
        """Average quality on the batch"""
        with torch.no_grad():
            output, _ = self._forward(batch._x, parameters, state, is_training=False)
        accuracy = torch.argmax(output, 1).eq(batch._y).sum().float() / len(batch)
        loss = self._criterion(output, batch._y)
        return {"loss": loss.item(), "accuracy": accuracy.item()}

    def evaluate(
        self,
        dataset: Dataset,
        parameters: List[torch.Tensor],
        state: List[torch.Tensor],
    ) -> Quality:
        """Average quality on a dataset"""
        mean_quality = None
        count = 0
        for _, batch in dataset.iterator(batch_size=250, shuffle=False, repeat=False):
            quality = self.quality(parameters, state, batch)
            if mean_quality is None:
                count = len(batch)
                mean_quality = quality
            else:
                count += len(batch)
                weight = float(len(batch)) / count
                for key, value in mean_quality.items():
                    mean_quality[key] += weight * (quality[key] - mean_quality[key])
        return mean_quality

    def _forward(
        self,
        input,
        parameters: List[torch.Tensor],
        state: List[torch.Tensor],
        is_training=False,
    ) -> Tuple[torch.Tensor, State]:
        if is_training:
            self._model.train()
        else:
            self._model.eval()

        for param, value in zip(self._model.parameters(), parameters):
            param.data = value

        for buffer, value in zip(self._model.buffers(), state):
            buffer.data = value

        output = self._model(input)
        state = [b.data for b in self._model.buffers()]

        return output, state

    def _create_model(self):
        if self._model_name == "ResNet_EvoNorm18":
            from .models.resnet_evonorm import ResNet_ImageNet

            model = ResNet_ImageNet(resnet_size=18)
            model.to(self._device)
            model.train()
        return model


class ImageNetDataset(PyTorchDataset):

    max_batch_size = 256

    def __init__(self, split, data_root=os.getenv("DATA"), device="cuda"):
        if split == "train":
            dataset = LMDBPT(
                os.path.join(data_root, "lmdb", "train.lmdb"),
                transform=get_transform("imagenet", augment=True, color_process=False),
                is_image=True,
            )
        elif split == "test":
            dataset = LMDBPT(
                os.path.join(data_root, "lmdb", "val.lmdb"),
                transform=get_transform("imagenet", augment=False, color_process=False),
                is_image=True,
            )
        else:
            raise ValueError(f"Unknown split '{split}'.")
        super().__init__(dataset, device=device)
