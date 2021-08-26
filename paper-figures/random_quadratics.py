import contextlib
import math

import torch


class RandomQuadraticsTask:
    def __init__(
        self, num_workers: int, sgd_noise: float, d: int, **kwargs
    ):
        self.d = d
        self.num_workers = num_workers
        self.sgd_noise_stdev = sgd_noise
        self.q = DistributedQuadraticsObjective(n=num_workers, d=d, **kwargs)

    def zeta2(self):
        return self.q.zeta2()

    def init_state(self):
        return torch.zeros(self.num_workers, self.d, 1)
    
    def grad(self, x):
        return self.q.grad(x, self.sgd_noise_stdev)

    def error(self, x):
        """Squared L2 norm / distance from the target"""
        # These behave similarly
        # return self.q.sq_distance_mean_to_optimum(x)
        return self.q.suboptimality_of_mean(x)

        # This behaves differently (good for D2 and RelaySum/Model, bad for Gossip and a bit bad for RelaySum/Grad)
        # return self.q.mean_sq_distance_to_optimum(x)


class DistributedQuadraticsObjective:
    """
    Random quadratics for testing distributed and decentralized learning algorithms.
    f(x) = avg_i ||A_i x - b_i||^2
    Allows you to control
    - number of workers `n`
    - dimensionality `d`
    - smoothness constant `L`
    - heterogeneity `zeta2`
    - strong convexity `mu`
    - initial distance from the optimum `r0` if initialized from 0
    """
    def __init__(self, n, d, L=1, heterogeneity=1, r0=1, seed=1, mu=None):
        assert heterogeneity >= 0
        assert L > 0
        assert n > 1
        if mu is not None:
            assert mu > 0 and mu < L

        self.n = n
        self.d = d

        with fork_rng_with_seed(seed):
            # Generate random quadratics, centered at zero
            self.A = torch.randn(n, d, d)
            self.B = torch.zeros(n, d, 1)

            # Add a stacked matrix view for convenience
            self.AA = self.A.view(n*d, d)
            self.BB = self.B.view(n*d, 1)

            # Make all of them L-smooth and mu-strongly-convex
            for a in self.A:
                if mu is None:
                    a.div_(torch.max(torch.svd(a).S))
                else:
                    U, S, V = torch.svd(a)
                    S = torch.linspace(mu, L, len(S))
                    a[:] = U @ (torch.diag(S) @ V.T)

            # Move the quadratics to have their own minima
            # Iteratively find the right scaling that results in the desired heterogeneity
            worker_optimum_offset_directions = torch.randn(n, d, 1)
            original_BB = self.BB.clone()
            scale = 1.0
            search_range_min = 0
            search_range_max = None
            cur_zeta2 = torch.tensor(0.0)
            while torch.abs(cur_zeta2 - heterogeneity) > 1e-5:
                # Move worker's optima in the right direction
                self.BB[:] = original_BB.clone()
                for a, b, opt in zip(self.A, self.B, worker_optimum_offset_directions):
                    b.add_(a @ (opt * scale))

                # Move the optimum back to zero
                self.BB.sub_(self.AA @ self._optimum()[0, :, :])
                self.optimum = self._optimum()
                cur_zeta2 = self.zeta2()

                # Binary search for `scale`
                if cur_zeta2 < heterogeneity and scale > search_range_min:
                    search_range_min = scale
                elif cur_zeta2 > heterogeneity and (search_range_max is None or scale < search_range_max):
                    search_range_max = scale
                if search_range_max is None:
                    scale *= 2
                else:
                    scale = (search_range_min + search_range_max) / 2

            # Move the optimum to a point at distance r0
            optimum = torch.randn(d, 1)
            optimum.mul_(r0 / optimum.norm())
            self.BB.add_(self.AA @ optimum)
            
        self.optimum = self._optimum()
        self.value_at_optimum = self(self._optimum())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # (1, d, 1) -> scalar
        return torch.sum((self.AA @ x - self.BB)**2) / self.n

    def suboptimality_of_mean(self, x: torch.Tensor) -> torch.Tensor:  # (n, d, 1) -> scalar
        mean_x = x.mean(0, keepdim=True)
        return self(mean_x) - self.value_at_optimum

    def mean_sq_distance_to_optimum(self, x: torch.Tensor) -> torch.Tensor:  # (n, d, 1) -> scalar
        return torch.sum((x - self.optimum)**2, dim=[1, 2]).mean()

    def sq_distance_mean_to_optimum(self, x: torch.Tensor) -> torch.Tensor:  # (n, d, 1) -> scalar
        mean_x = x.mean(0)
        optimum = self.optimum[0]
        return torch.sum((mean_x - optimum)**2)

    def local_losses(self, x: torch.Tensor) -> torch.Tensor:  # (n, d, 1) -> scalar
        return torch.sum((torch.einsum("nab, nbc -> nac", self.A, x) - self.B)**2, dim=[1,2])

    def grad(self, x: torch.Tensor, noise_stdev: float=0) -> torch.Tensor:  # (n, d, 1) -> (n, d, 1)
        # x: (n, d, 1)
        # out: (n, d, 1)
        AXmB = torch.einsum("nab,nbo->nao", self.A, x) - self.B
        grad = torch.einsum("njk,njf->nkf", self.A, AXmB)
        if noise_stdev > 0:
            grad.add_(torch.randn_like(x), alpha=noise_stdev / math.sqrt(self.d))
        return grad

    def zeta2(self) -> torch.Tensor:
        """Measure for heterogeneity"""
        g = self.grad(self.optimum)
        zeta2_individual = torch.sum(g**2, dim=[1, 2])  # size `n`
        return torch.mean(zeta2_individual)

    def smoothness(self) -> torch.Tensor:
        """Maximum (worst) smoothness of any individual worker"""
        return max(torch.max(torch.svd(a).S**2) for a in self.A)
    
    def r0(self) -> torch.Tensor:
        """Initial distance to the optimum if initialized at 0"""
        return torch.norm(self._optimum())
    
    def strong_convexity(self) -> torch.Tensor:
        """Maximum (worst) strong-convexity constant of any individual worker"""
        return min(torch.min(torch.svd(a).S**2) for a in self.A)
    
    def _optimum(self) -> torch.Tensor:
        opt, _ = torch.solve(self.AA.T @ self.BB, self.AA.T @ self.AA)
        return opt.unsqueeze(0)

    def _scale_to_smoothness(self, L=1) -> torch.Tensor:
        current_L = self.smoothness()
        self.A.data.mul_(torch.sqrt(L / current_L))
        self.B.data.mul_(torch.sqrt(L / current_L))
    
    def _ensure_minimum_strong_convexity(self, strong_convexity):
        for a in self.A:
            U, S, V = torch.svd(a)
            S = torch.cat([S[0:1], torch.minimum(S[1:], torch.tensor(math.sqrt(strong_convexity)))])
            a[:] = U @ (torch.diag(S) @ V.T)



class DistributedRandomLeastSquaresObjective:
    def __init__(self, num_batches, num_examples_per_batch, n, d, L=1, heterogeneity=1, r0=1, seed=1, mu=None):
        assert heterogeneity >= 0
        assert L > 0
        assert n > 1
        assert num_batches > 1
        assert num_batches / n  == num_batches // n

        if mu is not None:
            assert mu > 0 and mu < L

        nb = num_batches
        ne = num_examples_per_batch

        self.n = n
        self.d = d
        self.num_batches = num_batches
        self.num_examples_per_batch = num_examples_per_batch

        with fork_rng_with_seed(seed):
            # Generate a batch of random quadratics
            self.A = torch.randn(nb, ne, d)
            self.B = torch.zeros(nb, ne, 1)

            # Add a stacked matrix view for convenience
            self.AA = self.A.view(nb * ne, d)
            self.BB = self.B.view(nb * ne, 1)

            # And one stacked per worker
            self.worker_As = self.A.view(n, -1, d)
            self.worker_Bs = self.B.view(n, -1, 1)

            # Make all of them L-smooth and mu-strongly-convex
            for a in self.A:
                if mu is None:
                    a.div_(torch.max(torch.svd(a).S))
                else:
                    U, S, V = torch.svd(a)
                    S = torch.linspace(mu, L, len(S))
                    a[:] = U @ (torch.diag(S) @ V.T)

            # Move the quadratics to have their own minima
            # Iteratively find the right scaling that results in the desired heterogeneity
            worker_optimum_offset_directions = torch.randn(n, d, 1)
            original_BB = self.BB.clone()
            scale = 1.0
            search_range_min = 0
            search_range_max = None
            cur_zeta2 = torch.tensor(0.0)
            while torch.abs(cur_zeta2 - heterogeneity) > 1e-5:
                self.BB[:] = original_BB.clone()
                for a, b, opt in zip(self.A, self.B, worker_optimum_offset_directions):
                    b.add_(a @ (opt * scale))

                # Move the optimum back to zero
                self.BB.sub_(self.AA @ self._optimum()[0, :, :])
                self.optimum = self._optimum()
                cur_zeta2 = self.zeta2()

                # Binary search
                if cur_zeta2 < heterogeneity:
                    if scale > search_range_min:
                        search_range_min = scale
                elif cur_zeta2 > heterogeneity:
                    if search_range_max is None or scale < search_range_max:
                        search_range_max = scale
                if search_range_max is None:
                    scale *= 2
                else:
                    scale = (search_range_min + search_range_max) / 2

            # Move the optimum to a point at distance r0
            optimum = torch.randn(d, 1)
            optimum.mul_(r0 / optimum.norm())
            self.BB.add_(self.AA @ optimum)
            
        self.optimum = self._optimum()
        self.value_at_optimum = self(self._optimum())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # (1, d, 1) -> scalar
        return torch.sum((self.AA @ x - self.BB)**2) / (self.num_examples_per_batch * self.num_batches)

    def suboptimality_of_mean(self, x: torch.Tensor) -> torch.Tensor:  # (n, d, 1) -> scalar
        mean_x = x.mean(0, keepdim=True)
        return self(mean_x) - self.value_at_optimum

    def mean_sq_distance_to_optimum(self, x: torch.Tensor) -> torch.Tensor:  # (n, d, 1) -> scalar
        return torch.sum((x - self.optimum)**2, dim=[1, 2]).mean()

    def sq_distance_mean_to_optimum(self, x: torch.Tensor) -> torch.Tensor:  # (n, d, 1) -> scalar
        mean_x = x.mean(0)
        optimum = self.optimum[0]
        return torch.sum((mean_x - optimum)**2)

    def grad(self, x: torch.Tensor) -> torch.Tensor:  # (n, d, 1) -> (n, d, 1)
        # x: (n, d, 1)
        # out: (n, d, 1)
        AXmB = torch.einsum("nab,nbo->nao", self.worker_As, x) - self.worker_As
        grad = torch.einsum("njk,njf->nkf", self.worker_As, AXmB)
        num_examples_per_user = self.worker_As.shape[1]
        return grad / num_examples_per_user

    def stochastic_grad(self, x: torch.Tensor) -> torch.Tensor:  # (n, d, 1) -> (n, d, 1)
        # x: (n, d, 1)
        # out: (n, d, 1)
        num_examples_per_user = self.worker_As.shape[1]

        datapoints_per_user = torch.randint(high=num_examples_per_user, size=[self.n])
        user_indices = torch.arange(self.n)

        A_selection = self.worker_As[user_indices, datapoints_per_user]
        B_selection = self.worker_As[user_indices, datapoints_per_user]

        AXmB = torch.einsum("nab,nbo->nao", A_selection, x) - B_selection
        grad = torch.einsum("njk,njf->nkf", A_selection, AXmB)

        return grad

    def zeta2(self) -> torch.Tensor:
        """Measure for heterogeneity"""
        g = self.grad(self.optimum)
        zeta2_individual = torch.sum(g**2, dim=[1, 2])  # size `n`
        return torch.mean(zeta2_individual)

    def smoothness(self) -> torch.Tensor:
        return max(torch.max(torch.svd(a).S**2) for a in self.worker_As)
    
    def r0(self) -> torch.Tensor:
        """Initial distance to the optimum if initialized at 0"""
        return torch.norm(self._optimum())
    
    def strong_convexity(self) -> torch.Tensor:
        return min(torch.min(torch.svd(a).S**2) for a in self.worker_As)
    
    def _optimum(self) -> torch.Tensor:
        opt, _ = torch.solve(self.AA.T @ self.BB, self.AA.T @ self.AA)
        return opt.unsqueeze(0)


@contextlib.contextmanager
def fork_rng_with_seed(seed):
    if seed is None:
        yield
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield
