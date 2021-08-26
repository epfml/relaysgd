import math
import random
from typing import Any, Dict, Iterable, List, NamedTuple, Tuple

import torch
from torch.functional import norm

from base_optimizers import configure_base_optimizer
from tasks.api import Task
from topologies import configure_topology
from utils.communication import (
    MultiTopologyGossipMechanism,
    MultiTopologyRelayMechanism,
    get_rank,
    get_world_size,
    isend,
    num_bytes,
    pack,
    recv,
    unpack,
)
from utils.timer import Timer


class TrainStats(NamedTuple):
    step: int
    bytes_sent: int


class BatchStats(NamedTuple):
    loss: float


Parameters = List[torch.Tensor]


State = List[torch.Tensor]


def train(
    config: Dict[str, Any], task: Task, timer: Timer
) -> Iterable[Tuple[TrainStats, BatchStats, Parameters, State]]:
    if config["algorithm"] == "all-reduce":
        yield from allreduce(config, task, timer)
    elif config["algorithm"] == "gossip":
        yield from gossip(config, task, timer)
    elif config["algorithm"] == "relaysum-model":
        yield from relaysum_model(config, task, timer)
    elif config["algorithm"] == "relaysum-mix":
        yield from relaysum_mix(config, task, timer)
    elif config["algorithm"] == "relaysum-grad":
        yield from relaysum_grad(config, task, timer)
    elif config["algorithm"] == "d2":
        yield from d2(config, task, timer)
    elif config["algorithm"] == "gradient-tracking":
        yield from gradient_tracking(config, task, timer)
    elif config["algorithm"] == "quasi-global-momentum":
        yield from quasi_global_momentum(config, task, timer)
    elif config["algorithm"] == "push-sum":
        yield from push_sum(config, task, timer)
    else:
        raise ValueError("Unsupported algorithm {}".format(config["algorithm"]))


def allreduce(config, task: Task, timer: Timer):
    assert config["topology"] == "fully-connected"

    bytes_sent = 0
    last_loss = None

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)
    gradients = 0.0

    for step, batch in task.data.iterator(
        batch_size=config["batch_size"],
        shuffle=True,
        ref_num_data=task.mean_num_data_per_worker,
    ):
        timer.epoch = step
        yield (
            TrainStats(step, bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        if config["overlap_communication"] and step > 0:
            with timer("communication.send"):
                buffer, shapes = pack(gradients)
                comm_handle = torch.distributed.all_reduce(buffer, async_op=True)
                bytes_sent += num_bytes(buffer)

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )

        if config["overlap_communication"] and step > 0:
            with timer("communication.recv"):
                comm_handle.wait()
                buffer /= get_world_size()
                avg_gradients = unpack(buffer, shapes)

        if not config["overlap_communication"] or step > 0:
            with timer("local_update"):
                base_optimizer.step(
                    parameters,
                    avg_gradients if config["overlap_communication"] else gradients,
                    base_optimizer_state,
                    lr=config["learning_rate"] * learning_rate_schedule(config, step),
                )

        if not config["overlap_communication"]:
            with timer("communication"):
                buffer, shapes = pack(parameters)
                torch.distributed.all_reduce(buffer)
                buffer /= get_world_size()
                parameters = unpack(buffer, shapes)
                bytes_sent += num_bytes(buffer)


def gossip(config, task: Task, timer: Timer):
    last_loss = None

    topology = configure_topology(config)
    gossip = MultiTopologyGossipMechanism(topology, message_drop_prob=config["simulated_dropped_message_probability"])

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)

    for step, batch in task.data.iterator(
        batch_size=config["batch_size"],
        shuffle=True,
        ref_num_data=task.mean_num_data_per_worker,
    ):
        timer.epoch = step
        yield (
            TrainStats(step, gossip.bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        if config["overlap_communication"]:
            buffer, shapes = pack(parameters)
            gossip.send(buffer)
            del buffer

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )

        with timer("local_update"):
            base_optimizer.step(
                parameters,
                gradients,
                base_optimizer_state,
                lr=config["learning_rate"] * learning_rate_schedule(config, step),
            )

        with timer("communication"):
            buffer, shapes = pack(parameters)
            if not config["overlap_communication"]:
                gossip.send(buffer)
            gossip.gossip_update(buffer)
            parameters = unpack(buffer, shapes)


def relaysum_grad(config, task: Task, timer: Timer):
    """
    Currently with local momentum
    Possibly this would work better if you set the momentum buffer to the average update from before (like quasi-global)
    """
    last_loss = None

    topology = configure_topology(config)

    relay = MultiTopologyRelayMechanism(
        topology, overlap=config["overlap_communication"],
        message_drop_prob=config["simulated_dropped_message_probability"]
    )

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)

    for step, batch in task.data.iterator(
        batch_size=config["batch_size"],
        shuffle=True,
        ref_num_data=task.mean_num_data_per_worker,
    ):
        timer.epoch = step
        yield (
            TrainStats(step, relay.bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )
            updates = base_optimizer.compute_updates(
                parameters,
                gradients,
                base_optimizer_state,
                lr=config["learning_rate"] * learning_rate_schedule(config, step),
            )
            local_updates, shapes = pack(updates)

        with timer("communication"):
            relay.send(local_updates)
            avg_updates = relay.receive_average()

        with timer("local_update"):
            for p, u in zip(parameters, unpack(avg_updates, shapes)):
                p.data += u


def relaysum_model(config, task: Task, timer: Timer):
    last_loss = None

    topology = configure_topology(config)

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)

    initial_parameters, shapes = pack(parameters)  ## copies
    relay = MultiTopologyRelayMechanism(
        topology,
        overlap=config["overlap_communication"],
        initial_state=initial_parameters,
        message_drop_prob=config["simulated_dropped_message_probability"],
        normalization_mode=config.get("relaysum_model_normalization_mode", "world_size")
    )

    for step, batch in task.data.iterator(
        batch_size=config["batch_size"],
        shuffle=True,
        ref_num_data=task.mean_num_data_per_worker,
    ):
        timer.epoch = step
        yield (
            TrainStats(step, relay.bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )

        with timer("local_update"):
            base_optimizer.step(
                parameters,
                gradients,
                base_optimizer_state,
                lr=config["learning_rate"] * learning_rate_schedule(config, step),
            )

        with timer("communication.send"):
            buffer, shapes = pack(parameters)
            relay.send(buffer)
            buffer = relay.receive_average()
            parameters = unpack(buffer, shapes)

        if config.get("relaysum_model_update_ref_state", False):
            relay.set_initial_state(buffer)


def relaysum_mix(config, task: Task, timer: Timer):
    last_loss = None

    topology = configure_topology(config)

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)

    initial_parameters, shapes = pack(parameters)  ## copies
    relay = MultiTopologyRelayMechanism(
        topology,
        overlap=config["overlap_communication"],
        initial_state=initial_parameters,
        message_drop_prob=config["simulated_dropped_message_probability"]
    )

    for step, batch in task.data.iterator(
        batch_size=config["batch_size"],
        shuffle=True,
        ref_num_data=task.mean_num_data_per_worker,
    ):
        timer.epoch = step
        yield (
            TrainStats(step, relay.bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )

        with timer("local_update"):
            updates = base_optimizer.compute_updates(
                parameters,
                gradients,
                base_optimizer_state,
                lr=config["learning_rate"] * learning_rate_schedule(config, step),
            )

        with timer("communication"):
            param_buffer, shapes = pack(parameters)
            update_buffer, _ = pack(updates)
            relay.send(config["consensus_strength"] * param_buffer + update_buffer)

            param_buffer.mul_(1 - config["consensus_strength"])
            param_buffer.add_(relay.receive_average())
            parameters = unpack(param_buffer, shapes)


def gradient_tracking(config, task: Task, timer: Timer):
    assert not config["overlap_communication"]
    last_loss = None

    topology = configure_topology(config)
    gossip = MultiTopologyGossipMechanism(topology, message_drop_prob=config["simulated_dropped_message_probability"])
    correction_gossip = MultiTopologyGossipMechanism(topology, message_drop_prob=config["simulated_dropped_message_probability"])

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)
    correction = [torch.zeros_like(p) for p in parameters]

    for step, batch in task.data.iterator(
        batch_size=config["batch_size"],
        shuffle=True,
        ref_num_data=task.mean_num_data_per_worker,
    ):
        timer.epoch = step
        yield (
            TrainStats(step, gossip.bytes_sent + correction_gossip.bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )

        with timer("local_update"):
            prev_parameters = [p.clone() for p in parameters]
            base_optimizer.step(
                parameters,
                gradients,
                base_optimizer_state,
                lr=config["learning_rate"] * learning_rate_schedule(config, step),
            )
            updates = [p - prev for p, prev in zip(parameters, prev_parameters)]
            for p, c in zip(parameters, correction):
                p.add_(c)

        with timer("communication.parameters"):
            buffer, shapes = pack(parameters)
            gossip.send(buffer)
            gossip.gossip_update(buffer)
            parameters = unpack(buffer, shapes)

        with timer("communication.correction"):
            buffer, shapes = pack([c + u for c, u in zip(correction, updates)])
            correction_gossip.send(buffer)
            correction_gossip.gossip_update(buffer)
            correction = [b - u for b, u in zip(unpack(buffer, shapes), updates)]


def d2(config, task: Task, timer: Timer):
    assert not config["overlap_communication"]
    last_loss = None

    topology = configure_topology(config)
    gossip = MultiTopologyGossipMechanism(topology, message_drop_prob=config["simulated_dropped_message_probability"])

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)
    correction = [torch.zeros_like(p) for p in parameters]

    for step, batch in task.data.iterator(
        batch_size=config["batch_size"],
        shuffle=True,
        ref_num_data=task.mean_num_data_per_worker,
    ):
        timer.epoch = step
        yield (
            TrainStats(step, gossip.bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )

        with timer("local_update"):
            prev_parameters = [p.clone() for p in parameters]
            base_optimizer.step(
                parameters,
                gradients,
                base_optimizer_state,
                lr=config["learning_rate"] * learning_rate_schedule(config, step),
            )
            updates = [p - prev for p, prev in zip(parameters, prev_parameters)]
            for p, c in zip(parameters, correction):
                p.data.add_(c)

        with timer("communication"):
            buffer, shapes = pack(parameters)
            gossip.send(buffer)
            gossip.gossip_update(buffer)
            parameters = unpack(buffer, shapes)

        with timer("update_correction"):
            correction = [
                p - prev - u
                for p, prev, u in zip(unpack(buffer, shapes), prev_parameters, updates)
            ]


def quasi_global_momentum(config, task: Task, timer: Timer):
    assert not config["overlap_communication"]
    assert config["base_optimizer"] == "SGD"

    last_loss = None

    topology = configure_topology(config)
    gossip = MultiTopologyGossipMechanism(topology, message_drop_prob=config["simulated_dropped_message_probability"])

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)

    for step, batch in task.data.iterator(
        batch_size=config["batch_size"],
        shuffle=True,
        ref_num_data=task.mean_num_data_per_worker,
    ):
        timer.epoch = step
        yield (
            TrainStats(step, gossip.bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )

        lr = config["learning_rate"] * learning_rate_schedule(config, step)

        with timer("local_update"):
            prev_params = [p.data.clone() for p in parameters]
            prev_optimizer_state = [m.clone() for m in base_optimizer_state]
            base_optimizer.step(
                parameters,
                gradients,
                base_optimizer_state,
                lr=config["learning_rate"] * learning_rate_schedule(config, step),
            )
            base_optimizer_state = prev_optimizer_state  # restore

        with timer("communication"):  # parameters
            buffer, shapes = pack(parameters)
            gossip.send(buffer)
            gossip.gossip_update(buffer)
            parameters = unpack(buffer, shapes)

        with timer("update_momentum"):
            for m, p, prev in zip(base_optimizer_state, parameters, prev_params):
                m.mul_(config["momentum"]).add_(
                    prev - p, alpha=(1 - config["momentum"]) / max(lr, 1e-8)
                )


def push_sum(config, task: Task, timer: Timer):
    assert not config["overlap_communication"]
    bytes_sent = 0
    last_loss = None

    assert config["topology"] == "exponential"

    d = int(math.log2(get_world_size()))
    assert 2 ** d == get_world_size()

    parameters, state = task.initialize(seed=config["seed"])
    base_optimizer = configure_base_optimizer(config)
    base_optimizer_state = base_optimizer.init(parameters)

    for i, (step, batch) in enumerate(
        task.data.iterator(
            batch_size=config["batch_size"],
            shuffle=True,
            ref_num_data=task.mean_num_data_per_worker,
        )
    ):
        timer.epoch = step
        yield (
            TrainStats(step, bytes_sent),
            BatchStats(loss=last_loss),
            parameters,
            state,
        )

        with timer("compute_grad"):
            last_loss, gradients, state = task.loss_and_gradient(
                parameters, state, batch
            )

        with timer("local_update"):
            base_optimizer.step(
                parameters,
                gradients,
                base_optimizer_state,
                lr=config["learning_rate"] * learning_rate_schedule(config, step),
            )

        for j in range(config["push_sum_avg_steps"]):
            with timer("communication"):
                send_buffer, shapes = pack(parameters)

                offset = 2 ** (i * (config["push_sum_avg_steps"] + j) % d)
                n = get_world_size()

                # Send
                send_request_handles = []
                neighbor = int(get_rank() + offset) % n
                handle = isend(send_buffer, neighbor)
                bytes_sent += num_bytes(send_buffer)
                send_request_handles.append(handle)

                # Receive
                recv_buffer = torch.empty_like(send_buffer)
                neighbor = int(get_rank() - offset) % n
                recv(recv_buffer, neighbor)
                if random.uniform(0, 1) > config["simulated_dropped_message_probability"]:
                    avg_buffer = send_buffer * 0.5
                    avg_buffer.add_(recv_buffer, alpha=0.5)
                    for handle in send_request_handles:
                        handle.wait()
                    del send_buffer
                else:
                    avg_buffer = send_buffer
                    for handle in send_request_handles:
                        handle.wait()

                parameters = unpack(avg_buffer, shapes)


def learning_rate_schedule(config, epoch):
    """Apply any learning rate schedule"""
    lr = 1.0

    if config["distributed_world_size"] > 1 and config["num_lr_warmup_epochs"] > 0:
        warmup_epochs = config["num_lr_warmup_epochs"]
        max_factor = 1.0
        factor = 0 + (max_factor - 0) * min(epoch / warmup_epochs, 1.0)
        lr *= factor

    for (milestone, factor) in config["lr_schedule_milestones"]:
        if epoch >= milestone:
            lr *= factor
        else:
            return lr
    return lr
