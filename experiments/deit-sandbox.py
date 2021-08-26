#!/usr/bin/env python3

import os
import json
import subprocess

from shared import remote_exec, register_job, mongo, upload_code_package

code_package = upload_code_package()

gpus_per_node = 4
num_workers = 16

experiment = os.path.splitext(os.path.basename(__file__))[0]

description = "Just getting deit to work"

base_config = {
    "learning_rate": 0.1 * 32 * 16 / 512,  # lr * batch size * workers / 512
    "momentum": 0.9,
    "task": "DeIT",
    "num_lr_warmup_epochs": 0,
    "num_epochs": 300,
    "lr_schedule": "deit",
    "topology": "chain",
    "batch_size": 32,
    "test_interval": 4,
    "distributed_world_size": num_workers, 
    "gpus_per_node": gpus_per_node,
    "model_name": "deit_tiny_patch16_224",
    "non_iid_alpha": 1.0,
    "weight_decay": 0.05,
}

for algorithm, overlap, momentum in [
    ("all-reduce", False, 0.9),
]:
    config = {"algorithm": algorithm, "overlap_communication": overlap, "momentum": momentum}
    if algorithm == "all-reduce":
        config["topology"] = "fully-connected"
    elif algorithm == "relaysum-mix":
        config["consensus_strength"] = 0.5
    elif algorithm == "push-sum":
        config["topology"] = "exponential"

    job_name = f"{algorithm}-overlap{overlap}-mom{momentum}"
    if mongo.job.count_documents({"job": job_name, "experiment": experiment}) > 0:
        # We have this one already
        pass
    job_id = register_job(
        user="anonymized",
        project="average-routing",
        experiment=experiment,
        job=job_name,
        priority=10,
        n_workers=num_workers,
        config_overrides={**base_config, **config},
        runtime_environment={"clone": {"code_package": code_package}, "script": "train.py"},
        annotations={"description": description},
    )
    remote_exec(f'sbatch --nodes 1 --ntasks {num_workers} --gres gpu:{gpus_per_node} --cpus-per-task 2 --job-name="{job_name}" --wrap="srun jobrun {job_id} --mpi"')
