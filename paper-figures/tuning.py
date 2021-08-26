import sys

from torch._C import Value
sys.path.insert(0, "../../../code")

import math
import torch

from random_quadratics import RandomQuadraticsTask


def time_to_eps(method, topology, learning_rate, eps, max_steps=100000, seed=1, **args):
    torch.manual_seed(seed)
    
    task = RandomQuadraticsTask(topology.num_workers, seed=seed+10, **args)
    
    errors = [1000000] * 10  # some large number :p
    
    for iterate in method(task, topology, learning_rate, max_steps):
        is_time_to_test = (
            (iterate.step < 10) or 
            (iterate.step < 100 and iterate.step % 10 == 0) or 
            (iterate.step < 2000 and iterate.step % 50 == 0) or 
            (iterate.step < 10000 and iterate.step % 500 == 0) or 
            iterate.step % 1000 == 0
        )
        if is_time_to_test:
            error = task.error(iterate.state)
            errors.append(error.item())
            if torch.isnan(error):
                return ("diverged", error.item())
            elif len(errors) > 20 and min(errors[-10:]) > min(errors[:-10]) - eps:
                return ("not improving", error.item())
            else:
                errors.append(error.item())

            if all(e < eps for e in errors[-4:]):
                return ("reached", iterate.step)

    return ("reached max steps", error.item())


def best_time_to_eps(method, topology, eps, seed=1, max_steps=100000, **args):
    best_time = None
    best_lr = None

    learning_rates_tried = set()

    # Coarsely find a power-of-10 learning rate that reaches below eps
    lr_factor = 2
    lr = 1000
    res = None
    while res != "reached":
        # print(f"Trying {lr}")
        lr /= lr_factor
        res, value = time_to_eps(method, topology, lr, eps, max_steps=max_steps, seed=seed, **args)
        # print(f"- {res} {value}")
        learning_rates_tried.add(lr)

        if res == "reached" and (best_time is None or value < best_time):
            best_lr = lr
            best_time = value

        if res == "reached max steps":
            print(f"reached max steps {max_steps}")
            return None, None

    # Construct a grid around the learning rate found
    rnge = math.log10(lr_factor)
    for _ in range(4):
        grid = [v.item() for v in torch.logspace(math.log10(best_lr) - rnge, math.log10(best_lr) + rnge, 5) if v.item() not in learning_rates_tried]
        for lr in grid:
            res, value = time_to_eps(method, topology, lr, eps, max_steps=max_steps, seed=seed, **args)
            learning_rates_tried.add(lr)

            if res == "reached" and (best_time is None or value < best_time):
                best_lr = lr
                best_time = value

        if best_lr != min(learning_rates_tried) and best_lr != max(learning_rates_tried):
            rnge /= 2
    
    return best_time, best_lr


def curve_events(algorithm, task, topology, learning_rate, plateau_tolerance, max_steps, num_test_points, num_plateau_points=3):
    """
    Run an optimizer on a task, and emit events whenever something happens
    """
    errors = []
    
    test_iterates = set(torch.logspace(math.log2(1), math.log2(max_steps), num_test_points, base=2).round().int().unique().numpy())

    initial_error = None

    for iterate in algorithm(task, topology, learning_rate, max_steps):
        if iterate.step in test_iterates:
            error = task.error(iterate.state)
            errors.append(error)

            if initial_error is None:
                initial_error = error

            yield ("error", iterate.step, error.item())

            if torch.isnan(error) or error > initial_error:
                yield ("diverged", iterate.step, error.item())
                return

            if len(errors) >= 2 and error > errors[-2]:
                yield ("going up", iterate.step, error.item())

            if len(errors) >= num_plateau_points:
                last_errors = errors[-num_plateau_points:]
                mean_last_error = sum(last_errors) / num_plateau_points
                if all(abs(p - mean_last_error) < plateau_tolerance for p in last_errors):
                    yield ("reached plateau", iterate.step, mean_last_error.item())

    yield ("reached max steps", iterate.step, error.item())

def tune_plateau(start_lr, desired_plateau, verbose=False, **params):
    """
    Tune the plateau reached by a deterministic algorithm
    Don't allow oscillations
    """
    lr = start_lr

    search_range_min = None
    search_range_max = 2 * lr

    best_lr = None
    best_step = None

    i = 0
    while search_range_min is None or search_range_max - search_range_min > 1e-6:
        i += 1
        if i > 30:
            print("Too many iterations of tuning")
            return best_step, best_lr
        lr = search_range_max/2 if search_range_min is None else (search_range_min + search_range_max) / 2
        if verbose:
            print(f"Tryiing lr {lr}")
        for event, step, error in curve_events(**params, learning_rate=lr, plateau_tolerance=desired_plateau/100):
            if event == "diverged":
                if verbose:
                    print("-", event, step, error)
                search_range_max = max(search_range_min, lr) if search_range_min is not None else lr
                break
            elif event == "going up" and error > desired_plateau:
                if verbose:
                    print("-", event, step, error)
                search_range_max = max(search_range_min, lr) if search_range_min is not None else lr
                break
            elif event == "reached plateau":
                if verbose:
                    print("-", event, step, error)
                if error < desired_plateau and (best_step is None or step < best_step):
                    best_lr = lr
                    best_step = step

                if error < desired_plateau and error > desired_plateau / 1.1:
                    return best_step, best_lr
                elif error > desired_plateau:
                    search_range_max = max(lr, search_range_min) if search_range_min is not None else lr
                else:
                    search_range_min = min(lr, search_range_max)
                break
            elif event == "reached max steps":
                if verbose:
                    print("-", event, step, error)
                search_range_min = min(lr, search_range_max)
                break
            elif event == "error" and best_step is not None and step > best_step * 10:
                if verbose:
                    print("-", event, step, error)
                search_range_min = min(lr, search_range_max)
                break
            elif event == "error":
                pass
            elif event == "going up":
                pass
            else:
                raise ValueError(f"Unknown tuning event {event}")

    return best_step, best_lr


def tune_fastest(start_lr, target_quality, allow_going_up=False, verbose=False, **params):
    """
    Tune learning rate to reach `target_quality` as fast as possible
    Don't allow oscillations above target_quality
    """
    lr = start_lr

    search_range_min = None
    search_range_max = 2 * lr

    best_lr = None
    best_step = None

    i = 0

    lr = search_range_max
    while search_range_min is None or search_range_max - search_range_min > 1e-4:
        i += 1
        if i > 30:
            print("Too many iterations of tuning")
            return best_step, best_lr
        
        lr = lr if search_range_min is None else (search_range_min + search_range_max) / 2
        if verbose:
            print("lr", lr)

        for event, step, error in curve_events(**params, learning_rate=lr, plateau_tolerance=target_quality/10):
            if event == "diverged":
                if verbose:
                    print("-", event, step, error)
                search_range_max = max(search_range_min, lr) if search_range_min is not None else lr
                lr /= 2
                break
            elif event == "going up" and error > target_quality and not allow_going_up:
                if verbose:
                    print("-", event, step, error)
                search_range_max = max(search_range_min, lr) if search_range_min is not None else lr
                lr /= 2
                break
            elif error < target_quality:
                if verbose:
                    print("-", event, step, error)
                if best_step is None or step < best_step:
                    lr /= 2
                else:
                    search_range_min = min(lr, search_range_max)
                    search_range_max = min(lr * 4, search_range_max)
                if best_step is None or step < best_step:
                    best_step = step
                    best_lr = lr
                break
            elif event == "reached plateau":
                if verbose:
                    print("-", event, step, error)
                if error > target_quality:
                    search_range_max = max(lr, search_range_min) if search_range_min is not None else lr
                    lr /= 2
                break
            elif event == "reached max steps":
                if verbose:
                    print("-", event, step, error)
                search_range_min = min(lr, search_range_max)
                break
            elif event == "going up":
                pass
            elif event == "error" and best_step is not None and step > best_step * 10:
                if verbose:
                    print("-", event, step, error)
                search_range_min = min(lr, search_range_max)
                break
            elif event == "error":
                pass
            else:
                raise ValueError(f"Unknown tuning event {event}")

    return best_step, best_lr