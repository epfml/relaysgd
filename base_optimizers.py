from typing import Any, List, Mapping, NamedTuple
import torch


def configure_base_optimizer(config) -> "BaseOptimizer":
    if config["base_optimizer"] == "SGD":
        return SGD(config)
    elif config["base_optimizer"] == "Adam":
        return Adam(config)
    else:
        raise ValueError("Unknown base optimizer {}".format(config["base_optimizer"]))


OptimizerState = Mapping[str, Any]


class BaseOptimizer:
    def __init__(self, config):
        self.config = config

    def init(self, parameters: List[torch.Tensor]) -> OptimizerState:
        raise NotImplementedError()

    def step(
        self,
        parameters: List[torch.Tensor],
        gradients: List[torch.Tensor],
        optimizer_state: OptimizerState,
        lr: float,
    ) -> None:
        """Updates parameters and optimizer_state in place"""
        raise NotImplementedError()

    def compute_updates(
        self,
        parameters: List[torch.Tensor],
        gradients: List[torch.Tensor],
        optimizer_state: OptimizerState,
        lr: float,
    ) -> List[torch.Tensor]:
        """Updates optimizer_state in place, but returns update instead of updating parameters"""
        prev_parameters = [p.clone() for p in parameters]
        self.step(parameters, gradients, optimizer_state, lr)
        updates = [p - prev for p, prev in zip(parameters, prev_parameters)]
        for p, prev in zip(parameters, prev_parameters):
            p.data = prev
        return updates


class SGD(BaseOptimizer):
    def init(self, parameters: List[torch.Tensor]) -> OptimizerState:
        return [
            torch.zeros_like(p, memory_format=torch.preserve_format) for p in parameters
        ]

    def step(
        self,
        parameters: List[torch.Tensor],
        gradients: List[torch.Tensor],
        optimizer_state: OptimizerState,
        lr: float,
    ) -> None:
        """Updates parameters and optimizer_state in place"""
        torch.optim._functional.sgd(
            parameters,
            gradients,
            optimizer_state,
            weight_decay=0.0,  # already taken care of in the task
            momentum=self.config["momentum"],
            lr=lr,
            dampening=0.0,
            nesterov=True,
        )


class AdamState(NamedTuple):
    exp_avgs: List[torch.Tensor]
    exp_avg_sqs: List[torch.Tensor]
    max_exp_avg_sqs: List[torch.Tensor]
    step: List[int]


class Adam(BaseOptimizer):
    def init(self, parameters: List[torch.Tensor]) -> OptimizerState:
        return AdamState(
            [
                torch.zeros_like(p, memory_format=torch.preserve_format)
                for p in parameters
            ],
            [
                torch.zeros_like(p, memory_format=torch.preserve_format)
                for p in parameters
            ],
            [],
            [0 for p in parameters],
        )

    def step(
        self,
        parameters: List[torch.Tensor],
        gradients: List[torch.Tensor],
        optimizer_state: OptimizerState,
        lr: float,
    ) -> None:
        """Updates parameters and optimizer_state in place"""
        for i in range(len(optimizer_state.step)):
            optimizer_state.step[i] += 1
        torch.optim._functional.adam(
            parameters,
            gradients,
            optimizer_state.exp_avgs,
            optimizer_state.exp_avg_sqs,
            optimizer_state.max_exp_avg_sqs,
            # the default step starts from 0.
            state_steps=optimizer_state.step,
            amsgrad=False,
            beta1=0.9,
            beta2=0.999,
            lr=lr,
            weight_decay=0.0,  # already taken care of in the task
            eps=1e-8,
        )
