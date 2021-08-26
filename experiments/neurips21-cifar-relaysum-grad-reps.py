#!/usr/bin/env python3

import os
import json
import subprocess

from shared import remote_exec, register_job, mongo, upload_code_package

code_package = upload_code_package()

gpus_per_node = 4
num_workers = 16

experiment = os.path.splitext(os.path.basename(__file__))[0]

description = "Tuning RelaySum/Grad with and without momentum with randomly split data according to different values of alpha"

base_config = {
    "seed": 1,
    "task": "Cifar",
    "model_name": "VGG-11",
    "algorithm": "relaysum-grad",
    "overlap_communication": False,
    "base_optimizer": "SGD",
    "num_epochs": 200,
    "num_lr_warmup_epochs": 5,
    "lr_schedule_milestones": [(150, 0.1), (180, 0.1)],
    "batch_size": 32,
    "weight_decay": 1e-4,
    "data_split_method": "dirichlet",
    "non_iid_alpha": None,
    "distributed_world_size": num_workers, 
    "gpus_per_node": gpus_per_node,
}

best_lrs = {
    (1, "double-binary-trees", 0.0): 0.4,
    (.1, "double-binary-trees", 0.0): 0.4,
    (.01, "double-binary-trees", 0.0): 0.4,
    (1, "double-binary-trees", 0.9): 0.05/2,
    (.1, "double-binary-trees", 0.9): 0.05/2,
    (.01, "double-binary-trees", 0.9): 0.025/2,
    (1, "chain", 0.0): 0.4,
    (.1, "chain", 0.0): 0.4,
    (.01, "chain", 0.0): 0.4,
    (1, "chain", 0.9): 0.05,
    (.1, "chain", 0.9): 0.1,
    (.01, "chain", 0.9): 0.025,
}

for seed in [1, 2, 3]:
    for alpha in [1, 0.1, 0.01]:
        for topology in ["double-binary-trees"]:
            # with momentum
            lr = best_lrs[alpha, topology, 0.9]
            config = {**base_config, "learning_rate": lr, "momentum": 0.9, "topology": topology, "non_iid_alpha": alpha, "seed": seed}
            job_name = "alpha{non_iid_alpha}-{algorithm}-{topology}-mom{momentum}-lr{learning_rate}-{seed}".format(**config)
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

for seed in [1, 2, 3]:
    for alpha in [1, 0.1, 0.01]:
        for topology in ["double-binary-trees"]:
            # without momentum
            lr = best_lrs[alpha, topology, 0.0]
            config = {**base_config, "learning_rate": lr, "momentum": 0.0, "topology": topology, "non_iid_alpha": alpha, "seed": seed}
            job_name = "alpha{non_iid_alpha}-{algorithm}-{topology}-mom{momentum}-lr{learning_rate}-{seed}".format(**config)
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
