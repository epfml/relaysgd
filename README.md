# RelaySGD

[RelaySum for Decentralized Deep Learning on Heterogeneous Data](https://papers.nips.cc/paper/2021/file/ebbdfea212e3a756a1fded7b35578525-Paper.pdf)

Abstract: Because the workers only communicate with few neighbors without central coordination, these updates propagate progressively over the network.
This paradigm enables distributed training on networks without all-to-all connectivity, helping to protect data privacy as well as to reduce the communication cost of distributed training in data centers.
A key challenge, primarily in decentralized deep learning, remains the handling of differences between the workers' local data distributions.
To tackle this challenge, we introduce the RelaySum mechanism for information propagation in decentralized learning.
RelaySum uses spanning trees to distribute information exactly uniformly across all workers with finite delays depending on the distance between nodes.
In contrast, the typical gossip averaging mechanism only distributes data uniformly asymptotically while using the same communication volume per step as RelaySum.
We prove that RelaySGD, based on this mechanism, is independent of data heterogeneity and scales to many workers, enabling highly accurate decentralized deep learning on heterogeneous data.


## Assumed environment

- Python 3.7 from Anaconda 2021.05 (with numpy, pandas, matplotlib, seaborn)
- PyTorch 1.8.1
- NetworkX 2.4


We ran deep learning experiments with PyTorch Distributed using MPI. [environment/files/setup.sh](environment/files/setup.sh) describes our runtime environment and code we use to compile PyTorch with MPI support. For 16-worker Cifar-10 experiments, we used 4 Nvidia K80 GPUs on Google Cloud, each with 4 worker processes.

## Code organization

- The entrypoint for our deep learning code is [train.py](deep-learning/train.py).
- You manually need to start multiple instances of [train.py](deep-learning/train.py). This could be through MPI, Slurm or a simple script such as [dispatch.py](deep-learning/dispatch.py)
- It can run different experiments based on its global `config` variable. 
- All the `config`'s used in our experiments are listed in the scheduling code under [deep-learning/experiments](deep-learning/experiments).
- The __RelaySGD__ algorithm is implemented starting from line 225 in [algorithms.py](deep-learning/algorithms.py).
- The __RelaySum__ communication mechanism is in [utils/communication.py](utils/communication.py) from line 85.
- Hyperparameters (`config` overrides) used in our experiments can be found in the [experiments/](experiments) directory.

## Academic version of the algorithm

The real implementation of RelaySGD is in line 225 in [algorithms.py](deep-learning/algorithms.py). 
In the real version, the code represents a single worker, and communication is explicit.
Below, we include an ‘academic’ version of the algorithm that simulates all workers in a single process.

```python
from collections import defaultdict
from typing import Mapping, NamedTuple
import torch

def relaysgd(task, world, learning_rate, num_steps):
    state: torch.Tensor = task.init_state()  # shape [num_workers, ...]

    class Edge(NamedTuple):
        src: int
        dst: int

    # Initialize worker's memory, one entry for each edge in the network
    messages: Mapping[Edge, float] = defaultdict(float)  # default value 0.0
    counts: Mapping[Edge, int] = defaultdict(int)

    for step in range(num_steps):
        # Execute a model update on each worker
        g: torch.Tensor = task.grad(state)  # shape [num_workers, ...]
        state = state - learning_rate * g

        # Send messages
        new_messages = defaultdict(float)
        new_counts = defaultdict(int)

        for worker in world.workers:
            neighbors = world.neighbors(worker)
            for neighbor in neighbors:
                new_messages[worker, neighbor] = (
                    state[worker] +
                    sum(messages[n, worker] for n in neighbors if n != neighbor)
                )
                new_counts[worker, neighbor] = (
                    1 + 
                    sum(counts[n, worker] for n in neighbors if n != neighbor)
                )

        messages = new_messages
        counts = new_counts

        # Apply RelaySGD averaging
        for worker in world.workers:
            neighbors = world.neighbors(worker)
            num_messages = sum(counts[n, worker] for n in neighbors)
            state[worker] = (
                state[worker] * (world.num_workers - num_messages) 
                + sum(messages[n, worker] for n in neighbors)
            ) / world.num_workers
```

## Paper figures

The directory [paper-figures](paper-figures) contains scripts used to generate all tables and figures in the paper and appendix. 

A few files here that might be of interest: 
- [paper-figures/algorithms.py](paper-figures/algorithms.py) contains simulation code for decentralized learning on a single node. It has implementations of many algorithms (RelaySGD, RelaySGD/Grad, D2, DPSGD, Gradient tracking)
- [paper-figures/random_quadratics.py](paper-figures/random_quadratics.py) implements the synthetic functions we test the algorithms with (B.4)
- [paper-figures/tuning.py](paper-figures/tuning.py) contains the logic we use to automatically tune learning rates for experiments with random quadratics.

## Reference
If you use this code, please cite the following paper

```
@inproceedings{vogels2021relaysum,
  title={Relaysum for decentralized deep learning on heterogeneous data},
  author={Vogels, Thijs and He, Lie and Koloskova, Anastasia and Karimireddy, Sai Praneeth and Lin, Tao and Stich, Sebastian U and Jaggi, Martin},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```
