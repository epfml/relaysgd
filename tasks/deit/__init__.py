import os
import contextlib
import itertools
import re
from typing import Iterable, List, Tuple

import torch

from ..api import Batch, Dataset, Gradient, Loss, Parameters, Quality, State, Task
from ..cifar import PyTorchDataset, fork_rng_with_seed
from .datasets import build_dataset
import tasks.deit.models as models
from timm.models import create_model
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

class ImagenetTask(Task):
    def __init__(
        self, weight_decay, model_name, data_split_method, non_iid_alpha=None, seed=0, mixup=0.8, cutmix=1, cutmix_minmax=None, mixup_prob=1.0, mixup_switch_prob=0.5, mixup_mode="batch", smoothing=0.1
    ):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = ImagenetDataset("train", device=self._device)
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

        self._test_data = ImagenetDataset("test", device=self._device)

        self._model_name = model_name
        self._model = self._create_model()

        self._mixup_fn = None
        mixup_active = mixup > 0 or cutmix > 0. or cutmix_minmax is not None
        if mixup_active:
            self._mixup_fn = Mixup(
                mixup_alpha=mixup, cutmix_alpha=cutmix, cutmix_minmax=cutmix_minmax,
                prob=mixup_prob, switch_prob=mixup_switch_prob, mode=mixup_mode,
                label_smoothing=smoothing, num_classes=self._test_data.num_classes)

        self._criterion = LabelSmoothingCrossEntropy()
        if mixup > 0.:
            # smoothing is handled with mixup label transform
            self._criterion = SoftTargetCrossEntropy()
        elif smoothing:
            self._criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
        else:
            self._criterion = torch.nn.CrossEntropyLoss()


        self._weight_decay_per_param = [
            weight_decay
            for p, _ in self._model.named_parameters()
        ]

    def initialize(self, seed=42) -> Tuple[Parameters, State]:
        with fork_rng_with_seed(seed):
            self._model = self._create_model()
        parameters = [p.data for p in self._model.parameters()]
        state = [b.data for b in self._model.buffers()]
        return parameters, state

    def loss_and_gradient(
        self,
        parameters: List[torch.Tensor],
        state: List[torch.Tensor],
        batch: Batch,
        random_seed=None,
    ) -> Tuple[Loss, Gradient, State]:

        if self._mixup_fn is not None:
            samples, targets = self._mixup_fn(batch._x, batch._y)
        else:
            samples, targets = batch._x, batch._y

        with fork_rng_with_seed(random_seed):
            output, state = self._forward(samples, parameters, state, is_training=True)

        loss = self._criterion(output, targets)

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
        _, top5 = output.topk(5)
        top5_accuracy = batch._y.unsqueeze(1).eq(top5).sum().float() / len(batch)
        loss = torch.nn.CrossEntropyLoss().to(self._device)(output, batch._y)
        return {"loss": loss.item(), "accuracy": accuracy.item(), "top5_accuracy": top5_accuracy.item()}

    def evaluate(
        self,
        dataset: Dataset,
        parameters: List[torch.Tensor],
        state: List[torch.Tensor],
    ) -> Quality:
        """Average quality on a dataset"""
        mean_quality = None
        count = 0
        for _, batch in dataset.iterator(batch_size=200, shuffle=False, repeat=False):
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
        model = create_model(self._model_name, pretrained=False, num_classes=self._test_data.num_classes, drop_rate=0.0, drop_path_rate=0.1, drop_block_rate=None)
        model.to(self._device)
        model.train()
        return model


class ImagenetDataset(PyTorchDataset):
    max_batch_size = 64

    def __init__(self, split, data_root="/mnt/imagenet", device="cuda"):
        dataset, num_classes = build_dataset(is_train=split=="train", data_set="IMNET", data_path=data_root)
        super().__init__(dataset, device=device)
        self.num_classes = num_classes
