#!/usr/bin/env python3

import datetime
import math
import os

import numpy as np
import torch

from algorithms import train
from tasks.api import Task
from utils.accumulators import Mean as MeanAccumulator
from utils.communication import get_rank
from utils.timer import Timer

config = dict(
    seed=42,
    task="BERT-sst2",
    model_name="distilbert-base-uncased",
    # task="ImageNet",
    # model_name="ResNet_EvorNorm18",
    # task="ImageNet",
    # model_name="ResNet_EvoNorm18",
    # task="Cifar",
    # model_name="VGG-11",
    data_split_method="dirichlet",
    non_iid_alpha=1.0,
    num_epochs=200,
    batch_size=32,  # per worker
    algorithm="all-reduce",
    overlap_communication=False,
    topology="fully-connected",
    base_optimizer="Adam",
    learning_rate=1e-5,
    num_lr_warmup_epochs=5,
    lr_schedule_milestones=[(150, 0.1), (180, 0.1)],
    momentum=0.0,
    weight_decay=0.0001,
    test_interval=4,
    simulated_dropped_message_probability=0,
    log_verbosity=1,
    distributed_backend="mpi",
    distributed_rank=0,
    distributed_world_size=4,  # 1 = turn off
    distributed_init_file=None,
    gpus_per_node=1,
)

output_dir = "./output.tmp"  # can be overwritten by the code running this script


def main():
    torch.manual_seed(config["seed"] + config["distributed_rank"])
    np.random.seed(config["seed"] + config["distributed_rank"])

    timer = Timer(verbosity_level=config["log_verbosity"], log_fn=metric)

    init_distributed_pytorch()

    task = configure_task()

    epoch_metrics = MeanAccumulator()

    for train_stats, batch_stats, parameters, state in train(config, task, timer):
        epoch = train_stats.step
        training_time = {"epoch": epoch, "mb": train_stats.bytes_sent / 1024 / 1024}

        if batch_stats.loss is not None:
            epoch_metrics.add({"loss": batch_stats.loss})
            if math.isnan(batch_stats.loss):
                raise RuntimeError("diverged")

        if epoch % 1 == 0:
            info({"state.progress": epoch / config["num_epochs"]})

        if epoch % 1 == 0 and epoch > 0:
            with timer("report_train_stats"):
                epoch_metrics.reduce()  # average with other workers
                for key, value in epoch_metrics.value().items():
                    metric(
                        key, {"value": value, **training_time}, tags={"split": "train"}
                    )
            epoch_metrics.reset()

        if (epoch <= 5 and (epoch % 1 == 0)) or epoch % config["test_interval"] == 0:
            with timer("test"):
                test_stats = task.evaluate(task._test_data, parameters, state)
                for key, value in test_stats.items():
                    log_metric(
                        key,
                        {"value": value, **training_time},
                        tags={"split": "test", "worker": get_rank()},
                    )

                    # Check if nobody exceeded trivial accuracy.
                    if config["task"] == "Cifar" and key == "accuracy" and epoch == 20:
                        max_value = torch.tensor(value)
                        torch.distributed.all_reduce(max_value, torch.distributed.ReduceOp.MAX)
                        if max_value < 0.11:
                            raise RuntimeError("This is not going anywhere.")

        if (epoch <= 5 and (epoch % 1 == 0)) or epoch % config["test_interval"] == 0:
            for entry in timer.transcript():
                log_runtime(
                    entry["event"], entry["mean"], entry["std"], entry["instances"]
                )

        if epoch >= config["num_epochs"]:
            info({"state.progress": 1.0})
            break


def configure_task() -> Task:
    if config["task"] == "Cifar":
        from tasks.cifar import CifarTask, download

        if config["distributed_world_size"] > 1:
            if torch.distributed.get_rank() == 0:
                download()
            torch.distributed.barrier()

        return CifarTask(
            weight_decay=config["weight_decay"],
            model_name=config["model_name"],
            data_split_method=config["data_split_method"],
            non_iid_alpha=config["non_iid_alpha"],
            seed=config["seed"] + 100,
        )
    elif config["task"] == "ImageNet":
        from tasks.imagenet import ImageNetTask

        return ImageNetTask(
            weight_decay=config["weight_decay"],
            model_name=config["model_name"],
            data_split_method=config["data_split_method"],
            non_iid_alpha=config["non_iid_alpha"],
            seed=config["seed"] + 100,
        )
    elif config["task"] == "DeIT":
        from tasks.deit import ImagenetTask

        return ImagenetTask(
            weight_decay=config["weight_decay"],
            model_name=config["model_name"],
            data_split_method=config["data_split_method"],
            non_iid_alpha=config["non_iid_alpha"],
            seed=config["seed"] + 100,
        )
    elif "BERT" in config["task"]:
        from tasks.bert import BERTTask

        return BERTTask(
            weight_decay=config["weight_decay"],
            data_name=config["task"].split("-")[-1],
            model_name=config["model_name"],
            data_split_method=config["data_split_method"],
            non_iid_alpha=config["non_iid_alpha"],
            seed=config["seed"] + 100,
        )
    elif config["task"] == "Quadratics":
        from tasks.quadratics import QuadraticsTask

        return QuadraticsTask(
            d=config["quadratics_d"],
            non_iidness=config["quadratics_non_iidness"],
            sgd_noise_variance=config["quadratics_sgd_noise_variance"],
            seed=config["seed"] + 100,
        )
    elif config["task"] == "Delivery":
        from tasks.delivery import DeliveryTask

        return DeliveryTask()
    else:
        raise ValueError("Unsupported task {}".format(config["task"]))


def init_distributed_pytorch():
    if config["distributed_world_size"] > 1:
        if config["distributed_backend"] == "mpi":
            print("Initializing with MPI")
            torch.distributed.init_process_group("mpi")
            print(
                "Rank",
                torch.distributed.get_rank(),
                "world size",
                torch.distributed.get_world_size(),
            )
            torch.cuda.set_device(
                torch.distributed.get_rank() % config["gpus_per_node"]
            )
        else:
            if config["distributed_init_file"] is None:
                config["distributed_init_file"] = os.path.join(output_dir, "dist_init")
            print(
                "Distributed init: rank {}/{} - {}".format(
                    config["distributed_rank"],
                    config["distributed_world_size"],
                    config["distributed_init_file"],
                )
            )
            torch.distributed.init_process_group(
                backend=config["distributed_backend"],
                init_method="file://"
                + os.path.abspath(config["distributed_init_file"]),
                timeout=datetime.timedelta(seconds=120),
                world_size=config["distributed_world_size"],
                rank=config["distributed_rank"],
            )


def log_info(info_dict):
    """Add any information to MongoDB
    This function will be overwritten when called through run.py"""
    pass


def log_metric(name, values, tags={}):
    """Log timeseries data
    This function will be overwritten when called through run.py"""
    value_list = []
    for key in sorted(values.keys()):
        value = values[key]
        value_list.append(f"{key}:{value:7.3f}")
    values = ", ".join(value_list)
    tag_list = []
    for key, tag in tags.items():
        tag_list.append(f"{key}:{tag}")
    tags = ", ".join(tag_list)
    print("{name:30s} - {values} ({tags})".format(name=name, values=values, tags=tags))


def log_runtime(label, mean_time, std, instances):
    """This function will be overwritten when called through run.py"""
    pass


def info(*args, **kwargs):
    if config["distributed_rank"] == 0:
        log_info(*args, **kwargs)


def metric(*args, **kwargs):
    if config["distributed_rank"] == 0:
        log_metric(*args, **kwargs)


if __name__ == "__main__":
    main()
