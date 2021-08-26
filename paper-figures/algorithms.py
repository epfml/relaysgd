# %%
import math
from collections import defaultdict
from typing import Mapping, NamedTuple

import seaborn as sns
import torch

from topologies import *

"""
This file contains simulation code for several decentralized algorithms.
We use those in the random quadratic experiments
"""


# %%
sns.set_style("darkgrid")

# %%
def spectral_gap(matrix):
    abs_eigenvalues = sorted(torch.abs(torch.eig(matrix).eigenvalues[:, 0]))
    return abs_eigenvalues[-1] - abs_eigenvalues[-2]


#%%
class QuadraticsTask:
    """avg_i ||A_i x - b_i||^2"""

    def __init__(
        self, num_workers: int, d: float, zeta2: float, sgd_noise_variance: float = 0.0
    ):
        self.d = d
        self.zeta2 = zeta2
        self.sgd_noise_variance = sgd_noise_variance
        self.A = torch.stack(
            [
                1 / math.sqrt(num_workers) * torch.eye(d) * (i + 1)
                for i in range(0, num_workers)
            ]
        )
        self.B = torch.stack(
            [
                torch.randn(d, 1) * math.sqrt(zeta2) / (i + 1)
                for i in range(0, num_workers)
            ]
        )

        # Flatten all the parameters
        aa = self.A.view(-1, d)
        bb = self.B.view(-1, 1)
        self.target, _ = torch.solve(aa.T @ bb, aa.T @ aa)
        self.target = self.target.unsqueeze(0)

        self.num_workers = num_workers
        self.name = f"Non-idd quadratics (ζ^2={zeta2})"
        self.error_metric = "Mean L2 distance from optimum"

        def _grad(A, B, state, sigma):
            AXmB = torch.einsum("nab,nbo->nao", A, state) - B
            grad = torch.einsum("njk,njf->nkf", A, AXmB)
            grad += torch.randn_like(state) * sigma
            return grad
        self._grad = torch.jit.script(_grad)

    def init_state(self):
        return torch.zeros(self.num_workers, self.d, 1) + 1
    
    def grad(self, state):
        return self._grad(self.A, self.B, state, torch.tensor(math.sqrt(self.sgd_noise_variance / self.d)))

    def error(self, state):
        """Squared L2 norm / distance from the target"""
        return torch.mean((state - self.target) ** 2)



class OneMinusOneTask:
    def __init__(
        self, num_workers: int
    ):
        self.num_workers = num_workers
        self.name = f"One and minus one at ends of a strip"
        self.error_metric = "Avg. sq. L2 distance from optimum"

    def init_state(self):
        return torch.ones(self.num_workers, 1, 1) * torch.randn(size=[1,1,1])
        # return torch.randn(size=[self.num_workers,1,1])

    def grad(self, state):
        grad = torch.zeros_like(state)
        grad[:] = state
        grad[0] = state[0] - 1
        grad[self.num_workers - 1] = state[self.num_workers - 1] + 1
        return grad

    def error(self, state):
        """Mean Squared L2 norm"""
        return torch.mean(state ** 2)


class RandomConstantsTask:
    def __init__(
        self, num_workers: int, solution_difference: float, noise_scale: float
    ):
        self.num_workers = num_workers
        self.name = f"Random constants"
        self.error_metric = "Avg. sq. L2 distance from optimum"

        self.noise_scale = noise_scale

        self.targets = solution_difference * torch.randn(size=[num_workers, 1, 1], generator=torch.Generator().manual_seed(0))
        self.avg_target = self.targets.mean(dim=0, keepdims=True)

    def init_state(self):
        return torch.zeros_like(self.targets) + 1
        # return torch.randn(size=[self.num_workers,1,1])

    def grad(self, state):
        return state - self.targets + self.noise_scale * torch.randn_like(state)

    def error(self, state):
        """Mean Squared L2 norm"""
        return torch.mean((state - self.avg_target) ** 2)


class CircleTask:
    def __init__(
        self, num_workers: int, sigma: float, init_point = torch.tensor([0.,0.])
    ):
        self.num_workers = num_workers
        self.name = f"points on a 2D circle"
        self.d = 2
        self.error_metric = "Avg. sq. L2 distance from the target"

        self.sigma = sigma

        angles = torch.linspace(0, 2 * math.pi, self.num_workers)
        self.targets = (torch.stack([torch.cos(angles), torch.sin(angles)]).T).view(num_workers, 2, 1)

        self.init_point = init_point
        self.avg_target = self.targets.mean(dim=0, keepdims=True)

    def init_state(self):
        return self.init_point.clone().view(1, 2, 1) + 0 * self.targets

    def grad(self, state):
        return state - self.targets + self.sigma * torch.randn_like(state) / math.sqrt(2)

    def error(self, state):
        """Mean Squared L2 norm"""
        return torch.mean((state - self.avg_target) ** 2)


class Iterate(NamedTuple):
    step: int
    state: torch.Tensor
    num_messages_sent: int


class Edge(NamedTuple):
    src: int
    dst: int


def relaysum_grad(task, world, learning_rate, num_steps):
    state = task.init_state()

    # Initialize messages between connected workers to 0
    messages: Mapping[Edge, float] = defaultdict(float)

    num_messages_sent = 0

    for step in range(num_steps):
        yield Iterate(step, state, num_messages_sent)
        g = task.grad(state)

        new_messages = {}
        for worker in world.workers:
            neighbors = world.neighbors(worker)
            for neighbor in neighbors:
                new_messages[worker, neighbor] = g[worker] + sum(
                    messages[n, worker] for n in neighbors if n != neighbor
                )

        for worker in world.workers:
            neighbors = world.neighbors(worker)
            sum_grads = g[worker] + sum(new_messages[n, worker] for n in neighbors)

            state[worker] -= learning_rate * sum_grads / world.num_workers

        messages = new_messages

        num_messages_sent += world.max_degree

    yield Iterate(step, state, num_messages_sent)


def relaysum_grad_overlap(task, world, learning_rate, num_steps, momentum=0):
    state = task.init_state()

    # Initialize messages between connected workers to 0
    messages: Mapping[Edge, float] = defaultdict(float)

    num_messages_sent = 0

    momentum_buffer = torch.zeros_like(state)

    for step in range(num_steps):
        yield Iterate(step, state, num_messages_sent)
        g = task.grad(state)
        momentum_buffer.mul_(momentum).add_(g, alpha=(1-momentum))

        # lr = min(step / 100, 1) * learning_rate
        lr = learning_rate
        # lr = min(step / 100, 1) * learning_rate
        # if step > num_steps // 2:
        #     lr /= 2

        new_messages = {}
        for worker in world.workers:
            neighbors = world.neighbors(worker)
            for neighbor in neighbors:
                new_messages[worker, neighbor] = lr * momentum_buffer[worker] + sum(
                    messages[n, worker] for n in neighbors if n != neighbor
                )

            sum_grads = lr * momentum_buffer[worker] + sum(messages[n, worker] for n in neighbors)

            # momentum_buffer[worker] = sum_grads / world.num_workers

            state[worker] -= sum_grads / world.num_workers

        messages = new_messages

        num_messages_sent += world.max_degree

    yield Iterate(step, state, num_messages_sent)





def relaysum_grad_star(task, world, learning_rate, num_steps, momentum=0):
    state = task.init_state()

    # Initialize messages between connected workers to 0
    messages: Mapping[Edge, float] = defaultdict(float)
    new_messages: Mapping[Edge, float] = defaultdict(float)

    num_messages_sent = 0

    momentum_buffer = torch.zeros_like(state)

    for step in range(num_steps):
        yield Iterate(step, state, num_messages_sent)
        g = task.grad(state)
        momentum_buffer.mul_(momentum).add_(g, alpha=(1-momentum))

        lr = learning_rate

        new_messages = {}
        for worker in world.workers:
            neighbors = world.neighbors(worker)
            for neighbor in neighbors:
                new_messages[worker, neighbor] = lr * momentum_buffer[worker] + sum(
                    messages[n, worker] for n in neighbors if n != neighbor
                )

            sum_grads = lr * momentum_buffer[worker] + sum(messages[n, worker] for n in neighbors)

            # momentum_buffer[worker] = sum_grads / world.num_workers

            state[worker] -= sum_grads / world.num_workers

        messages = new_messages

        num_messages_sent += world.max_degree

    yield Iterate(step, state, num_messages_sent)



def relaysum_mix(task, world, learning_rate, num_steps, momentum=0, alpha=0.5):
    state = task.init_state()

    init_state = state[0].clone()

    messages: Mapping[Edge, float] = defaultdict(float)

    num_messages_sent = 0

    momentum_buffer = torch.zeros_like(state)

    for step in range(num_steps):
        yield Iterate(step, state, num_messages_sent)
        g = task.grad(state)
        momentum_buffer.mul_(momentum).add_(g, alpha=(1-momentum))

        new_messages = {}
        for worker in world.workers:
            to_send = (1-alpha)*state[worker] - learning_rate * momentum_buffer[worker]
            neighbors = world.neighbors(worker)
            for neighbor in neighbors:
                new_messages[worker, neighbor, "content"] = to_send + sum(
                    messages[n, worker, "content"] for n in neighbors if n != neighbor
                )
                new_messages[worker, neighbor, "count"] = 1 + sum(
                    messages[n, worker, "count"] for n in neighbors if n != neighbor
                )

            sum_grads = to_send + sum(messages[n, worker, "content"] for n in neighbors)
            num_messages = 1 + sum(messages[n, worker, "count"] for n in neighbors)

            state[worker] = (
                alpha * state[worker] + 
                    (sum_grads + (1-alpha)*init_state * (world.num_workers - num_messages)) / world.num_workers
            )

            # This also works, if you remove `- learning_rate * momentum_buffer[worker]` from messages
            # state[worker] = alpha * (state[worker] - sum_grads / world.num_workers) + \
            #                 (1-alpha) * ((state[worker] + sum(messages[n, worker, "algo1"] for n in neighbors) + init_state * (world.num_workers - num_messages)) / world.num_workers)

        messages = new_messages

        num_messages_sent += world.max_degree

    yield Iterate(step, state, num_messages_sent)


def relaysum_model_overlap(task, world, learning_rate, num_steps, momentum=0):
    state = task.init_state()

    init_state = state[0].clone()

    # Initialize messages between connected workers to 0
    messages: Mapping[Edge, float] = defaultdict(float)
    counts: Mapping[Edge, int] = defaultdict(int)

    num_messages_sent = 0
    
    momentum_buffer = torch.zeros_like(state)

    for step in range(num_steps):
        yield Iterate(step, state, num_messages_sent)
        g = task.grad(state)
        momentum_buffer.mul_(momentum).add_(g, alpha=(1-momentum))

        # state -= learning_rate * ((momentum_buffer * momentum/(1-momentum)) + g)
        state -= learning_rate * momentum_buffer

        new_messages = {}
        new_counts = {}
        for worker in world.workers:
            neighbors = world.neighbors(worker)
            for neighbor in neighbors:
                new_messages[worker, neighbor] = state[worker] + sum(
                    messages[n, worker] for n in neighbors if n != neighbor
                )
                new_counts[worker, neighbor] = 1 + sum(
                    counts[n, worker] for n in neighbors if n != neighbor
                )

            num_messages = 1 + sum(counts[n, worker] for n in neighbors)

            state[worker] = (
                state[worker] + sum(messages[n, worker] for n in neighbors) + init_state * (world.num_workers - num_messages)
            ) / world.num_workers

            # state[worker] = (
            #     state[worker] + sum(messages[n, worker] for n in neighbors)
            # ) / num_messages

        messages = new_messages
        counts = new_counts

        num_messages_sent += world.max_degree

    yield Iterate(step, state, num_messages_sent)


def relaysum_model(task, world, learning_rate, num_steps):
    state = task.init_state()

    init_state = state[0].clone()

    # Initialize messages between connected workers to 0
    messages: Mapping[Edge, float] = defaultdict(float)
    counts: Mapping[Edge, int] = defaultdict(int)

    num_messages_sent = 0
    for step in range(num_steps):
        yield Iterate(step, state, num_messages_sent)
        g = task.grad(state)

        state -= learning_rate * g

        new_messages = defaultdict(float)
        new_counts = defaultdict(int)
        for worker in world.workers:
            neighbors = world.neighbors(worker)
            for neighbor in neighbors:
                new_messages[worker, neighbor] = state[worker] + sum(
                    messages[n, worker] for n in neighbors if n != neighbor
                )
                new_counts[worker, neighbor] = 1 + sum(
                    counts[n, worker] for n in neighbors if n != neighbor
                )

        for worker in world.workers:
            neighbors = world.neighbors(worker)
            num_messages = 1 + sum(new_counts[n, worker] for n in neighbors)
            state[worker] = (
                state[worker] + sum(new_messages[n, worker] for n in neighbors) + init_state * (world.num_workers - num_messages)
            ) / world.num_workers

        messages = new_messages
        counts = new_counts

        num_messages_sent += world.max_degree

    yield Iterate(step, state, num_messages_sent)


def time_varying_sgp(task, world, learning_rate, num_steps, overlap=False, num_communication_rounds=2):
    state = task.init_state()

    d = int(math.log2(world.num_workers-1)) + 1

    num_messages_sent = 0
    i = 0
    for step in range(num_steps):
        yield Iterate(step, state, num_messages_sent)
        g = task.grad(state)

        if not overlap:
            state -= learning_rate * g

        for _ in range(num_communication_rounds):
            shift = 2 ** (i % d)
            state = (torch.roll(state, shift, dims=0) + state) / 2
            i += 1
        
        num_messages_sent += 1

        if overlap:
            state -= learning_rate * g

    yield Iterate(step, state, num_messages_sent)



def gossip(
    task, world, learning_rate, num_steps, overlap=False, gossip_weight=None, momentum=0
):
    state = task.init_state()

    W = world.gossip_matrix(gossip_weight)

    momentum_buffer = torch.zeros_like(state)

    num_messages_sent = 0
    for step in range(num_steps):
        yield Iterate(step, state, num_messages_sent)
        g = task.grad(state)

        momentum_buffer.mul_(momentum).add_(g, alpha=(1-momentum))

        if not overlap:
            state -= learning_rate * momentum_buffer

        # gossip
        state = torch.einsum("wmn, wv -> vmn", state, W)

        num_messages_sent += world.max_degree

        if overlap:
            state -= learning_rate * momentum_buffer

    yield Iterate(step, state, num_messages_sent)


def all_reduce(
    task, world, learning_rate, num_steps, overlap=False, momentum=0
):
    state = task.init_state()

    momentum_buffer = torch.zeros_like(state)

    num_messages_sent = 0
    prev_g = 0.0
    for step in range(num_steps):
        yield Iterate(step, state, num_messages_sent)
        if overlap:
            g = prev_g
            prev_g = task.grad(state)
        else:
            g = task.grad(state)

        momentum_buffer.mul_(momentum).add_(g, alpha=(1-momentum))

        state -= learning_rate * momentum_buffer

        # averaging
        state = torch.mean(state, dim=0, keepdims=True).tile([state.shape[0], 1, 1])

        num_messages_sent += world.max_degree


    yield Iterate(step, state, num_messages_sent)


def quasi_global_momentum(
    task, world, learning_rate, num_steps, overlap=False, gossip_weight=None, momentum=0.9
):
    state = task.init_state()

    W = world.gossip_matrix(gossip_weight)

    momentum_buffer = torch.zeros_like(state)

    num_messages_sent = 0
    for step in range(num_steps):
        yield Iterate(step, state, num_messages_sent)
        g = task.grad(state)

        prev_state = state.clone()

        if not overlap:
            state -= learning_rate * (momentum_buffer * momentum + g) / (1+momentum)

        # gossip
        state = torch.einsum("wmn, wv -> vmn", state, W)

        momentum_buffer.mul_(momentum).add_((prev_state - state) / learning_rate, alpha=(1-momentum))

        num_messages_sent += world.max_degree

        if overlap:
            state -= learning_rate * (momentum_buffer * momentum + g)

    yield Iterate(step, state, num_messages_sent)


def gradient_tracking(task, world, learning_rate, num_steps, gossip_weight=None):
    state = task.init_state()

    W = world.gossip_matrix(gossip_weight)

    correction = torch.zeros_like(state)

    num_messages_sent = 0
    for step in range(num_steps):
        yield Iterate(step, state, num_messages_sent)
        g = task.grad(state)

        state -= learning_rate * (g - correction)

        # gossip
        state = torch.einsum("wmn, wv -> vmn", state, W)
        correction = torch.einsum("wmn, wv -> vmn", correction - g, W) + g

        num_messages_sent += 2 * world.max_degree

    yield Iterate(step, state, num_messages_sent)


def d2(task, world, learning_rate, num_steps, gossip_weight=None):
    state = task.init_state()

    W = world.gossip_matrix(gossip_weight)

    correction = torch.zeros_like(state)

    num_messages_sent = 0
    for step in range(num_steps):
        yield Iterate(step, state, num_messages_sent)
        g = task.grad(state)


        update = -learning_rate * g
        prev_state = state.clone()
        state = torch.einsum("wmn, wv -> vmn", state + update + correction, W)
        correction = state - prev_state - update

        num_messages_sent += world.max_degree

    yield Iterate(step, state, num_messages_sent)


def exact_diffusion(task, world, learning_rate, num_steps, gossip_weight=None):
    """
    This differs from D2 by using a different gossip matrix (W+I)/2
    """
    state = task.init_state()

    W = (world.gossip_matrix(gossip_weight) + torch.eye(world.num_workers))/2

    correction = torch.zeros_like(state)

    num_messages_sent = 0
    for step in range(num_steps):
        yield Iterate(step, state, num_messages_sent)
        g = task.grad(state)

        update = -learning_rate * g
        prev_state = state.clone()
        state = torch.einsum("wmn, wv -> vmn", state + update + correction, W)
        correction = state - prev_state - update

        num_messages_sent += world.max_degree

    yield Iterate(step, state, num_messages_sent)
