#!/usr/bin/env python3

import os
import json
import subprocess

from shared import remote_exec, register_job, mongo, upload_code_package

code_package = upload_code_package()

gpus_per_node = 4
num_workers = 16

experiment = os.path.splitext(os.path.basename(__file__))[0]

description = "Testing if we can tollerate randomly dropping messages"

base_config = {
    "seed": 1,
    "task": "Cifar",
    "model_name": "VGG-11",
    "algorithm": "relaysum-model",
    "overlap_communication": False,
    "base_optimizer": "SGD",
    "num_epochs": 200,
    "num_lr_warmup_epochs": 5,
    "lr_schedule_milestones": [(150, 0.1), (180, 0.1)],
    "batch_size": 32,
    "learning_rate": 0.3,
    "momentum": 0.9,
    "relaysum_model_normalization_mode": "world_size",
    "relaysum_model_update_ref_state": True,
    "topology": "double-binary-trees",
    "weight_decay": 1e-4,
    "data_split_method": "dirichlet",
    "non_iid_alpha": 0.01,
    "distributed_world_size": num_workers, 
    "gpus_per_node": gpus_per_node,
}

for drop_prob in [0, 0.1, 0.01, 0.001]:
    config = {**base_config, "simulated_dropped_message_probability": drop_prob}
    job_name = "{algorithm}-{learning_rate}-drop{simulated_dropped_message_probability}-v2".format(**config)
    if mongo.job.count_documents({"job": job_name, "experiment": experiment, **{f"config.{key}": value for key, value in config.items()}}) > 0:
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
