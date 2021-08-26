#!/usr/bin/env python3

import os
import json
import subprocess

from shared import remote_exec, register_job, mongo, upload_code_package

code_package = upload_code_package()

gpus_per_node = 4
num_workers = 16

experiment = os.path.splitext(os.path.basename(__file__))[0]

description = "Running this after refactoring as a unit test"

base_config = {
    "learning_rate": 1,
    "momentum": 0,
    "task": "Delivery",
    "num_lr_warmup_epochs": 0,
    "num_epochs": 100,
    "lr_schedule_milestones": [],
    "topology": "chain",
    "batch_size": 1,
    "test_interval": 1,
    "distributed_world_size": num_workers, 
    "gpus_per_node": gpus_per_node,
    "model_name": None,
    "non_iid_alpha": None,
    "weight_decay": None,
}


for algorithm, overlap, momentum in [
    # ("relaysum-model", False, 0.9),
    # ("all-reduce", False, 0.9),
    # ("relaysum-model", True, 0),
    # ("relaysum-model", False, 0),
    # ("relaysum-grad", True, 0),
    # ("relaysum-mix", False, 0),
    # ("all-reduce", False, 0),
    ("gossip", False, 0),
    ("gossip", True, 0),
    # ("push-sum", False, 0),
    # ("d2", False, 0),
    # ("gradient-tracking", False, 0),
    # ("quasi-global-momentum", False, 0.9)
]:
    config = {**base_config, "algorithm": algorithm, "overlap_communication": overlap, "momentum": momentum}
    if algorithm == "all-reduce":
        config["topology"] = "fully-connected"
    elif algorithm == "relaysum-mix":
        config["consensus_strength"] = 0.5
    elif algorithm == "push-sum":
        config["topology"] = "exponential"
        config["push_sum_avg_steps"] = 1

    topology = config["topology"]
    job_name = f"{algorithm}-overlap{overlap}-mom{momentum}-{topology}"
    if mongo.job.count_documents({"job": job_name, "experiment": experiment}) > 0:
        # We have this one already
        continue
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
