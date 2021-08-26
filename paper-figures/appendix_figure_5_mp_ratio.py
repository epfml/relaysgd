import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt

from topologies import (BinaryTreeTopology, ChainTopology, StarTopology,
                        Topology)


def construct_matrix(world: Topology):
    """
    For a chain of length 3, this would output:
    tensor([[0.3333, 0.0000, 0.0000, 0.0000, 0.3333, 0.0000, 0.0000, 0.0000, 0.3333],
            [0.0000, 0.3333, 0.0000, 0.3333, 0.0000, 0.3333, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.3333, 0.0000, 0.3333, 0.0000, 0.3333, 0.0000, 0.0000],
            [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000]])
    """
    g = world.to_networkx()
    distances = dict(nx.all_pairs_shortest_path_length(g))
    
    tau = world.max_delay
    n = world.num_workers

    # matrix of shape (tau*n) * (tau*n)
    M = torch.zeros(tau+1, n, tau+1, n)

    # add permutation of blocks to 
    for delay in range(0, tau):
        M[delay+1, :, delay, :] = torch.eye(n)

    # add the averaging entries
    for worker in world.workers:
        for neighbor in world.workers:
            M[0, worker, distances[worker][neighbor], neighbor] = 1/n

    # Make it into a matrix form
    M = M.view((tau+1)*n, (tau+1)*n)

    assert M.sum(1).allclose(torch.ones([]))

    return M

def compute_p_eigenvalue(world: Topology):
    W = construct_matrix(world)

    # # raise to a high power
    # onePi = W.clone()
    # for _ in range(20):
    #     onePi = onePi @ onePi

    eigenvalues = torch.view_as_complex(torch.eig(W).eigenvalues).abs()
    second_largest_eigenvalue = sorted(eigenvalues)[-2]
    return 1 - second_largest_eigenvalue**2

def compute_m_p(world: Topology):
    W = construct_matrix(world)

    converged = W.clone()
    for _ in range(20):
        converged = converged @ converged

    second_eigenvalue = sorted(torch.view_as_complex(torch.eig(W).eigenvalues).abs())[-2]

    def spectral_norm_sq(m):
        return max(torch.svd(m).S)**2

    init_sn = spectral_norm_sq(W - converged)

    best_m_over_p = None
    bad_counter = 0
    for m in range(1, 100):
        ratio = spectral_norm_sq(torch.matrix_power(W, m) - converged) / init_sn
        if ratio < 1:
            p = 1 - ratio**(1/(2*m))
            if p < second_eigenvalue:
                if best_m_over_p is None or m/p < best_m_over_p:
                    best_m_over_p = m/p
                    best_m = m
                    best_p = p
                else:
                    bad_counter += 1
                    if bad_counter > 10:
                        break
    return best_m, best_p, best_m_over_p

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    import seaborn as sns
    sns.set_theme("paper")
    sns.set_style("whitegrid")
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        'text.latex.preamble' : r'\usepackage{amsmath}\usepackage{newtxmath}'
    })
    sns.set_palette([sns.color_palette("tab10")[i] for i in [0, 1, 3, 9, 7]])

    # fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False)
    fig, ax1 = plt.subplots(figsize=(4, 3))

    for topo_name, topo in (("tree", BinaryTreeTopology), ("chain", ChainTopology), ("star", StarTopology)):
        nn = [2, 3, 4, 6, 8, 12, 16, 24, 32] #, 48, 64]
        mm_pp_ratio = [compute_m_p(topo(n)) for n in nn]
        mm = np.array([x[0] for x in mm_pp_ratio])
        pp = np.array([x[1] for x in mm_pp_ratio])
        ratios = np.array([x[2] for x in mm_pp_ratio])
        ns = np.array([topo(n).num_workers for n in nn])

        ax1.plot(ns, ratios, label=topo_name)
        # ax2.plot(ns, mm, label=topo_name)

    ax1.set_xlabel("Number of workers")
    # ax2.set_xlabel("Number of workers")
    # ax1.set_xlabel("Maximum delay")
    # ax2.set_xlabel("Maximum delay")
    ax1.set_ylabel(r"Smallest $\rho = \frac{m}{p}$ that satisfies the Lemma")
    # ax2.set_ylabel("m")
    ax1.legend()
    # ax2.legend()
    # plt.show()
    plt.tight_layout()
    fig.savefig("../writing/neurips21/figures/m-p-ratios.pdf", bbox_inches="tight")
