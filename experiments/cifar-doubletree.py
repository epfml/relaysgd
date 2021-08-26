#!/usr/bin/env python3

import os
import json
import subprocess

from shared import remote_exec, register_job, mongo, upload_code_package

code_package = upload_code_package()

gpus_per_node = 4
num_workers = 16

experiment = os.path.splitext(os.path.basename(__file__))[0]

description = "Playing with double binary trees"

base_config = {
    "learning_rate": 0.16,
    "momentum": 0.9,
    "task": "Cifar",
    "num_lr_warmup_epochs": 5,
    "topology": "binary-tree",
    "batch_size": 32,
    "distributed_world_size": num_workers, 
    "gpus_per_node": gpus_per_node,
    "non_iid_alpha": 0.1,
    "weight_decay": 1e-4,
}

for algorithm, overlap, momentum in [
    # ("gossip", False, 0),
    ("relaysum-model", False, 0.9),
    ("relaysum-grad", True, 0),
]:
    config = {**base_config, "algorithm": algorithm, "overlap_communication": overlap, "momentum": momentum}
    if algorithm == "all-reduce":
        config["topology"] = "fully-connected"
    if algorithm == "relaysum-mix":
        config["consensus_strength"] = 0.5
    if algorithm.startswith("push-sum"):
        config["topology"] = "exponential"
    if algorithm == "relaysum-model":
        config["learning_rate"] *= 1
    if algorithm == "quasi-global-momentum":
        config["learning_rate"] /= 4
    if algorithm == "push-sum":
        config["push_sum_avg_steps"] = 1
    if algorithm == "push-sum-2":
        config["algorithm"] = "push-sum"
        config["push_sum_avg_steps"] = 2
    if algorithm == "push-sum-3":
        config["algorithm"] = "push-sum"
        config["push_sum_avg_steps"] = 3
    # if momentum == 0:
    #     config["learning_rate"] /= 2

    topology = config["topology"]
    job_name = f"{algorithm}-overlap{overlap}-mom{momentum}-{topology}"
    if mongo.job.count_documents({"job": job_name, "experiment": experiment, **{f"config.{key}": value for key, value in config.items()}}) > 0:
        # We have this one already
        pass
    job_id = register_job(
        user="anonymized",
        project="average-routing",
        experiment=experiment,
        job=job_name,
        priority=10,
        n_workers=num_workers,
        config_overrides=config,
        runtime_environment={"clone": {"code_package": code_package}, "script": "train.py"},
        annotations={"description": description},
    )
    remote_exec(f'sbatch --nodes 1 --ntasks {num_workers} --gres gpu:{gpus_per_node} --cpus-per-task 2 --job-name="{job_name}" --wrap="srun jobrun {job_id} --mpi"')
