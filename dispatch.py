#!/usr/bin/env python3

"""
This runs multiple copies of train.py in parallel, for each worker
"""

import sys
from contextlib import contextmanager
import multiprocessing

config = dict(
    seed=42,
    task="ImageNet",
    model_name="ResNet_EvoNorm18",
    data_split_method="dirichlet",
    non_iid_alpha=1.0,
    num_epochs=200,
    batch_size=32,  # per worker
    algorithm="gossip",
    overlap_communication=True,
    topology="ring",
    base_optimizer="SGD",
    learning_rate=0.16,
    num_lr_warmup_epochs=5,
    lr_schedule_milestones=[(150, 0.1), (180, 0.1)],
    momentum=0.0,
    weight_decay=0.0001,
    test_interval=4,
    log_verbosity=1,
    distributed_backend="gloo",
    distributed_rank=0,
    distributed_world_size=4,  # 1 = turn off
    distributed_init_file=None,
    gpus_per_node=1,
)


def worker(rank):
    import train

    # Override config from train.py
    for key in list(train.config.keys()):
        del train.config[key]
    for key, value in config.items():
        train.config[key] = value

    train.config["distributed_rank"] = rank

    train.output_dir = output_dir
    train.log_metric = log_metric
    train.log_info = log_info
    train.log_runtime = log_runtime

    with print_prefix(f"Worker {rank}"):
        train.main()


def main():
    num_workers = config["distributed_world_size"]

    processes = [
        multiprocessing.Process(target=worker, args=(i,)) for i in range(num_workers)
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()


output_dir = "output.tmp"


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


@contextmanager
def print_prefix(prefix):
    global is_new_line
    orig_write = sys.stdout.write
    is_new_line = True

    def new_write(*args, **kwargs):
        global is_new_line
        if args[0] == "\n":
            is_new_line = True
        elif is_new_line:
            orig_write("[" + str(prefix) + "]: ")
            is_new_line = False
        orig_write(*args, **kwargs)

    sys.stdout.write = new_write
    yield
    sys.stdout.write = orig_write


def log_runtime(label, mean_time, std, instances):
    """This function will be overwritten when called through run.py"""
    pass


if __name__ == "__main__":
    main()
