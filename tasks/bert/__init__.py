import os
import types
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

from ..api import Batch, Dataset, Gradient, Loss, Parameters, Quality, State, Task
from ..cifar import PyTorchDataset, parameter_type, fork_rng_with_seed

from .linear_predictors import ptl2classes
from .task_configs import task2dataiter
from ..utils.non_iid_dirichlet import distribute_data_dirichlet


class BERTTask(Task):
    def __init__(
        self,
        weight_decay,
        model_name,
        data_name,
        data_split_method,
        non_iid_alpha=None,
        seed=0,
    ):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._pretrained_lm = model_name.split("-")[0]
        self._model_name = model_name

        self._data_name = data_name
        self._data = BERTDataset(
            model_name,
            pretrained_lm=self._pretrained_lm,
            data_name=data_name,
            split="train",
            device=self._device,
        )
        self.max_batch_size = self._data.max_batch_size
        self._test_data = BERTDataset(
            model_name,
            pretrained_lm=self._pretrained_lm,
            data_name=data_name,
            split="test",
            device=self._device,
        )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Splitting data by worker
            num_workers = torch.distributed.get_world_size()
            if data_split_method == "dirichlet":
                splits = self._data.dirichlet_split(
                    num_workers, non_iid_alpha, seed=seed
                )
            elif data_split_method == "random":
                splits = self._data.random_split(
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
            self.data = self._data
            self.mean_num_data_per_worker = len(self._data)

        self._model = self._create_model()
        self._criterion = torch.nn.CrossEntropyLoss().to(self._device)

        self._weight_decay_per_param = [
            0 if parameter_type(p) == "batch_norm" else weight_decay
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
        with fork_rng_with_seed(random_seed):
            output, state = self._forward(
                batch._x, parameters=parameters, state=state, is_training=True
            )
        loss = self._criterion(output, batch._y)
        gradients = torch.autograd.grad(loss, list(self._model.parameters()))
        gradients = [
            g + wd * p
            for g, wd, p in zip(gradients, self._weight_decay_per_param, parameters)
        ]
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
        batched,
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
            buffer.data = value.clone()

        output, *_ = self._model(**batched)
        state = [b.data for b in self._model.buffers()]

        return output, state

    def _create_model(self):
        classes = ptl2classes[self._pretrained_lm]
        pretrained_weight_path = os.path.join(os.getenv("DATA"), "pretrained_weights")

        # define dataset types.
        vector_cls_sentence_datasets = [
            "mrpc",
            "sst2",
            "mnli",
            "qqp",
            "cola",
            "qnli",
            "rte",
            "agnews",
            "trec",
            "dbpedia",
            "yelp2",
            "semeval16",
            "germeval",
            "imdb",
        ]
        postagging_datasets = ["posptb", "conll2003"]
        multiplechoice_datasets = ["swag"]

        # define model.
        if self._data_name in vector_cls_sentence_datasets:
            model = classes.seqcls.from_pretrained(
                self._model_name,
                num_labels=self._data.data_iter.num_labels,
                cache_dir=pretrained_weight_path,
            )
        elif self._data_name in postagging_datasets:
            model = classes.postag.from_pretrained(
                self._model_name,
                out_dim=self._data.data_iter.num_labels,
                cache_dir=pretrained_weight_path,
            )
        elif self._data_name in multiplechoice_datasets:
            model = classes.multiplechoice.from_pretrained(
                self._model_name, cache_dir=pretrained_weight_path
            )

        model.to(self._device)
        model.train()

        for param_name, param in model.named_parameters():
            if self._pretrained_lm in param_name:
                param.requires_grad = True
            if "classifier" in param_name:
                param.requires_grad = True
        return model


"""classes for dataset."""


class BERTDataset(PyTorchDataset):
    max_batch_size = 256

    def __init__(
        self,
        model_name,
        pretrained_lm,
        data_name,
        split,
        data_root=os.getenv("DATA"),
        device="cuda",
        max_sequence_len=200,
    ):
        # create data_iter.
        self.pretrained_lm = pretrained_lm
        classes = ptl2classes[pretrained_lm]
        tokenizer = classes.tokenizer.from_pretrained(model_name)
        data_iter = task2dataiter[data_name](
            data_name, model_name, tokenizer, max_sequence_len
        )

        dataset = data_iter.trn_dl if split == "train" else data_iter.val_dl
        self.data_iter = data_iter
        self._batch_to_device = types.MethodType(
            task2batched_fn[self.data_iter.task], self
        )
        super().__init__(dataset, device=device, prepare_batch=self.prepare_batch)

    def dirichlet_split(
        self,
        num_workers: int,
        alpha: float = 1,
        seed: int = 0,
        distribute_evenly: bool = True,
    ) -> List[Dataset]:
        indices_per_worker = distribute_data_dirichlet(
            self._set.golds, alpha, num_workers, num_auxiliary_workers=10, seed=seed
        )

        if distribute_evenly:
            indices_per_worker = np.array_split(
                np.concatenate(indices_per_worker), num_workers
            )

        return [
            PyTorchDataset(
                Subset(self._set, indices),
                device=self._device,
                prepare_batch=self.prepare_batch,
            )
            for indices in indices_per_worker
        ]

    def prepare_batch(self, batch):
        uids, golds, batched, _ = self._batch_to_device(batch)
        if (
            self.pretrained_lm == "roberta" or self.pretrained_lm == "distilbert"
        ) and "token_type_ids" in batched:
            batched.pop("token_type_ids")
        return Batch(batched, golds).to(self._device)


"""functions for batch_to_device."""


def seqcls_batch_to_device(self, batched):
    uids = batched[0]
    input_ids, golds, attention_mask, token_type_ids = batched[1:]
    return (
        uids,
        golds,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
        None,
    )


def tagging_batch_to_device(self, batched):
    uids = batched[0]
    input_ids, attention_mask, _golds, if_tgts = batched[1:]

    golds = []
    for b_step in range(_golds.shape[0]):
        gold = _golds[b_step][if_tgts[b_step]]
        golds.append(gold)

    if self.conf.bert_conf_["task"] != "conll2003":
        return (
            uids,
            torch.cat(golds, dim=0),
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "if_tgts": if_tgts,
            },
            None,
        )
    return (
        uids,
        torch.cat(golds, dim=0),
        {"input_ids": input_ids, "attention_mask": attention_mask, "if_tgts": if_tgts},
        _golds,
    )


task2batched_fn = {
    "mrpc": seqcls_batch_to_device,
    "sst2": seqcls_batch_to_device,
    "mnli": seqcls_batch_to_device,
    "qqp": seqcls_batch_to_device,
    "cola": seqcls_batch_to_device,
    "qnli": seqcls_batch_to_device,
    "rte": seqcls_batch_to_device,
    "posptb": tagging_batch_to_device,
    "swag": seqcls_batch_to_device,
    "agnews": seqcls_batch_to_device,
    "trec": seqcls_batch_to_device,
    "dbpedia": seqcls_batch_to_device,
    "yelp2": seqcls_batch_to_device,
    "semeval16": seqcls_batch_to_device,
    "conll2003": tagging_batch_to_device,
}
