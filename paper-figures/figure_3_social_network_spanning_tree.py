#%%
# %load_ext autoreload
# %autoreload 2
#%%
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt

import random_quadratics
import tuning
from algorithms import d2, exact_diffusion, gossip, relaysum_model
from topologies import *
from warehouse import Warehouse

torch.set_default_dtype(torch.float64)

sns.set_theme("paper")
sns.set_style("whitegrid")
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    'text.latex.preamble' : r'\usepackage{amsmath}\usepackage{newtxmath}'
})

# %%
warehouse = Warehouse()
warehouse_comm = Warehouse()
# %%
world = SocialNetworkTopology()
trees = [SocialNetworkTreeTopology(root) for root in world.workers]
task_no_noise = random_quadratics.RandomQuadraticsTask(world.num_workers, sgd_noise=0.0, d=10, heterogeneity=0.1)
task_noisy = random_quadratics.RandomQuadraticsTask(world.num_workers, sgd_noise=0.1, d=10, heterogeneity=0.1)

# %%
_, lr_gossip = tuning.tune_plateau(start_lr=1, desired_plateau=1e-5, task=task_no_noise, algorithm=gossip, topology=world, max_steps=20000, num_test_points=1000)

# %%
_, lr_gossip_tree = tuning.tune_plateau(start_lr=1, desired_plateau=1e-5, task=task_no_noise, algorithm=gossip, topology=trees[0], max_steps=20000, num_test_points=1000)

# %%
# _, lr_relaysum_grad = tuning.tune_plateau(start_lr=1, desired_plateau=1e-5, task=task_no_noise, algorithm=relaysum_grad, topology=trees[0], max_steps=2000, num_test_points=1000)


def d2_mod(*args, **kwargs):
    return d2(*args, **kwargs, gossip_weight = 0.08)

# %%
_, lr_d2 = tuning.tune_fastest(start_lr=1, target_quality=1e-5, task=task_no_noise, algorithm=exact_diffusion, topology=world, max_steps=2000, num_test_points=1000, allow_going_up=True)

# %%
_, lr_d2_tree = tuning.tune_fastest(start_lr=1, target_quality=1e-5, task=task_no_noise, algorithm=exact_diffusion, topology=trees[0], max_steps=2000, num_test_points=1000)

# %%
_, lr_relaysum_model = tuning.tune_fastest(start_lr=10, target_quality=1e-5, task=task_no_noise, algorithm=relaysum_model, topology=trees[0], max_steps=2000, num_test_points=1000)

# %%
learning_rate = 0.1
num_steps = 600
eval_interval = num_steps // 100

# task = task_no_noise
# noise = 0
# world = SocialNetworkTopology()
# for algorithm_name, algorithm in (("DP-SGD on social network", gossip),):
#     tags = dict(
#         algorithm=algorithm_name,
#         root="any",
#         noise=noise
#     )
#     warehouse.clear("error", tags)
#     torch.manual_seed(1)
#     for iterate in algorithm(task, world, learning_rate=lr_gossip, num_steps=num_steps, overlap=False):
#         if iterate.step % eval_interval == 0:
#             error = task.error(iterate.state).item()
#             warehouse.log_metric(
#                 "error", {"value": error, "step": iterate.step}, tags
#             )

# world = SocialNetworkTopology()
# for algorithm_name, algorithm in (("D2 on social network", d2_mod),):
#     tags = dict(
#         algorithm=algorithm_name,
#         root="any",
#         noise=noise
#     )
#     warehouse.clear("error", tags)
#     torch.manual_seed(1)
#     for iterate in algorithm(task, world, learning_rate=lr_d2, num_steps=num_steps,):
#         if iterate.step % eval_interval == 0:
#             error = task.error(iterate.state).item()
#             warehouse.log_metric(
#                 "error", {"value": error, "step": iterate.step}, tags
#             )

# for root in range(32):
#     world = SocialNetworkTreeTopology(root)
#     for algorithm_name, algorithm in (("DP-SGD on spanning trees", gossip),):
#         tags = dict(
#             algorithm=algorithm_name,
#             root=root,
#             noise=noise
#         )
#         warehouse.clear("error", tags)
#         torch.manual_seed(1)
#         for iterate in algorithm(task, world, learning_rate=lr_gossip_tree, num_steps=num_steps, overlap=False):
#             if iterate.step % eval_interval == 0:
#                 error = task.error(iterate.state).item()
#                 warehouse.log_metric(
#                     "error", {"value": error, "step": iterate.step}, tags
#                 )

# for root in range(32):
#     world = SocialNetworkTreeTopology(root)
#     for algorithm_name, algorithm in (("RelaySum/Model", relaysum_model),):
#         tags = dict(
#             algorithm=algorithm_name,
#             root=root,
#             noise=noise
#         )
#         warehouse.clear("error", tags)
#         torch.manual_seed(1)
#         for iterate in algorithm(task, world, learning_rate=lr_relaysum_model, num_steps=num_steps):
#             if iterate.step % eval_interval == 0:
#                 error = task.error(iterate.state).item()
#                 warehouse.log_metric(
#                     "error", {"value": error, "step": iterate.step}, tags
#                 )

# for root in range(32):
#     world = SocialNetworkTreeTopology(root)
#     for algorithm_name, algorithm in (("D2 on spanning trees", exact_diffusion),):
#         tags = dict(
#             algorithm=algorithm_name,
#             root=root,
#             noise=noise
#         )
#         warehouse.clear("error", tags)
#         torch.manual_seed(1)
#         for iterate in algorithm(task, world, learning_rate=lr_d2_tree, num_steps=num_steps):
#             if iterate.step % eval_interval == 0:
#                 error = task.error(iterate.state).item()
#                 warehouse.log_metric(
#                     "error", {"value": error, "step": iterate.step}, tags
#                 )

#%%
learning_rate = 0.1
num_steps = 500
eval_interval = num_steps // 100

#%%
task = task_noisy
noise = 1
world = SocialNetworkTopology()
for algorithm_name, algorithm in (("DP-SGD on social network", gossip),):
    tags = dict(
        algorithm=algorithm_name,
        root="any",
        noise=noise
    )
    warehouse.clear("error", tags)
    torch.manual_seed(1)
    amount_communicated = 0
    for iterate in algorithm(task, world, learning_rate=lr_gossip/1.1, num_steps=num_steps, overlap=False):
        if iterate.step % eval_interval == 0:
            error = task.error(iterate.state).item()
            warehouse.log_metric(
                "error", {"value": error, "step": iterate.step}, tags
            )
            warehouse_comm.log_metric(
                "error", {"value": error, "step": amount_communicated}, tags
            )
        amount_communicated += 2 * len(world.to_networkx().edges)

#%%
world = SocialNetworkTopology()
for algorithm_name, algorithm in (("D2 on social network", exact_diffusion),):
    tags = dict(
        algorithm=algorithm_name,
        root="any",
        noise=noise
    )
    warehouse.clear("error", tags)
    torch.manual_seed(1)
    amount_communicated = 0
    for iterate in algorithm(task, world, learning_rate=0.1, num_steps=num_steps,):
        if iterate.step % eval_interval == 0:
            error = task.error(iterate.state).item()
            warehouse.log_metric(
                "error", {"value": error, "step": iterate.step}, tags
            )
            warehouse_comm.log_metric(
                "error", {"value": error, "step": amount_communicated}, tags
            )
        amount_communicated += 2 * len(world.to_networkx().edges)

# for root in range(32):
#     world = SocialNetworkTreeTopology(root)
#     for algorithm_name, algorithm in (("DP-SGD on spanning trees", gossip),):
#         tags = dict(
#             algorithm=algorithm_name,
#             root=root,
#             noise=noise
#         )
#         warehouse.clear("error", tags)
#         torch.manual_seed(1)
#         amount_communicated = 0
#         for iterate in algorithm(task, world, learning_rate=lr_gossip_tree, num_steps=num_steps*3, overlap=False):
#             if iterate.step % eval_interval == 0:
#                 error = task.error(iterate.state).item()
#                 warehouse.log_metric(
#                     "error", {"value": error, "step": iterate.step}, tags
#                 )
#                 warehouse_comm.log_metric(
#                     "error", {"value": error, "step": amount_communicated}, tags
#                 )
#             amount_communicated += 2 * len(world.to_networkx().edges)

# for root in range(32):
#     world = SocialNetworkTreeTopology(root)
#     for algorithm_name, algorithm in (("D2 on spanning trees", exact_diffusion),):
#         tags = dict(
#             algorithm=algorithm_name,
#             root=root,
#             noise=noise
#         )
#         warehouse.clear("error", tags)
#         torch.manual_seed(1)
#         amount_communicated = 0
#         for iterate in algorithm(task, world, learning_rate=min(0.1, lr_d2_tree), num_steps=num_steps*3):
#             if iterate.step % eval_interval == 0:
#                 error = task.error(iterate.state).item()
#                 warehouse.log_metric(
#                     "error", {"value": error, "step": iterate.step}, tags
#                 )
#                 warehouse_comm.log_metric(
#                     "error", {"value": error, "step": amount_communicated}, tags
#                 )
#             amount_communicated += 2 * len(world.to_networkx().edges)

#%%
warehouse.clear("error", dict(algorithm="RelaySum/Grad", noise=1))
warehouse.clear("error", dict(algorithm="RelaySum/Model", noise=1))
for root in range(32):
    world = SocialNetworkTreeTopology(root)
    # lr = 0.1
    # for algorithm_name, algorithm in (("RelaySum/Grad", relaysum_grad),):
    #     tags = dict(
    #         algorithm=algorithm_name,
    #         root=root,
    #         noise=noise
    #     )
    #     warehouse.clear("error", tags)
    #     torch.manual_seed(1)
    #     for iterate in algorithm(task, world, learning_rate=lr, num_steps=num_steps):
    #         if iterate.step % eval_interval == 0:
    #             error = task.error(iterate.state).item()
    #             warehouse.log_metric(
    #                 "error", {"value": error, "step": iterate.step}, tags
    #             )
    # _, lr = tuning.tune_fastest(start_lr=1, target_quality=1e-5, task=task_noisy, algorithm=relaysum_model, topology=world, max_steps=2000, num_test_points=1000)
    
    lr = 0.3
    for algorithm_name, algorithm in (("RelaySum/Model", relaysum_model),):
        tags = dict(
            algorithm=algorithm_name,
            root=root,
            noise=noise
        )
        warehouse.clear("error", tags)
        torch.manual_seed(1)
        amount_communicated = 0
        for iterate in algorithm(task, world, learning_rate=lr, num_steps=num_steps*3):
            if iterate.step % eval_interval == 0:
                error = task.error(iterate.state).item()
                warehouse.log_metric(
                    "error", {"value": error, "step": iterate.step}, tags
                )
                warehouse_comm.log_metric(
                    "error", {"value": error, "step": amount_communicated}, tags
                )
            amount_communicated += 2 * len(world.to_networkx().edges)


# %%
# sns.lineplot()
colors = sns.color_palette("tab10")

f, [ax0, ax1] = plt.subplots(ncols=2, figsize=(5, 2.5))

ax=ax0
dt = warehouse.query("error", {"algorithm": "DP-SGD on social network", "noise": 1})
sns.lineplot(x="step", y="value", data=dt, label="DP-SGD on full network", color=colors[0], ax=ax, linestyle="--")

dt = warehouse.query("error", {"algorithm": "D2 on social network", "noise": 1})
sns.lineplot(x="step", y="value", data=dt, label="D2 on full network", color=colors[1], ax=ax, linestyle="--")

# for i in range(32):
#     dt = warehouse.query("error", {"algorithm": "DP-SGD on spanning trees", "root": i, "noise": 1})
#     sns.lineplot(x="step", y="value", data=dt, label="DP-SGD on spanning trees" if i == 0 else None, color=colors[0], ax=ax, alpha=0.3)

# for i in range(32):
#     dt = warehouse.query("error", {"algorithm": "D2 on spanning trees", "root": i, "noise": 1})
#     if len(dt) > 0:
#         sns.lineplot(x="step", y="value", data=dt, label="D2 on spanning trees" if i == 0 else None, color=colors[1], ax=ax, alpha=0.3)

for i in range(32):
    dt = warehouse.query("error", {"algorithm": "RelaySum/Model", "root": i, "noise": 1})
    if len(dt) > 0:
        sns.lineplot(x="step", y="value", data=dt, label="RelaySum/Model on spanning trees" if i == 0 else None, color=colors[3], ax=ax, alpha=0.3)
ax.set(yscale="log")

ax.set(xlim=[0,num_steps])
ax.set(ylim=[1e-6, 1])

ax.set_ylabel(r"Suboptimality $f(\bar{\mathbf{x}}) - f(\mathbf{x}^\star)$")
ax.set_xlabel("Steps")
ax.get_legend().remove()
# ax.set_title(r"$\sigma^2=0$ (w/o stochastic noise)")


ax=ax1
dt = warehouse_comm.query("error", {"algorithm": "DP-SGD on social network", "noise": 1})
sns.lineplot(x="step", y="value", data=dt, label="DP-SGD on full network", color=colors[0], ax=ax, linestyle="--")

dt = warehouse_comm.query("error", {"algorithm": "D2 on social network", "noise": 1})
sns.lineplot(x="step", y="value", data=dt, label="D2 on full network", color=colors[1], ax=ax, linestyle="--")

# for i in range(32):
#     dt = warehouse_comm.query("error", {"algorithm": "DP-SGD on spanning trees", "root": i, "noise": 1})
#     sns.lineplot(x="step", y="value", data=dt, label="DP-SGD on spanning trees" if i == 0 else None, color=colors[0], ax=ax, alpha=0.3)

# for i in range(32):
#     dt = warehouse_comm.query("error", {"algorithm": "D2 on spanning trees", "root": i, "noise": 1})
#     if len(dt) > 0:
#         sns.lineplot(x="step", y="value", data=dt, label="D2 on spanning trees" if i == 0 else None, color=colors[1], ax=ax, alpha=0.3)

for i in range(32):
    dt = warehouse_comm.query("error", {"algorithm": "RelaySum/Model", "root": i, "noise": 1})
    if len(dt) > 0:
        sns.lineplot(x="step", y="value", data=dt, label="RelaySum/Model on spanning trees" if i == 0 else None, color=colors[3], ax=ax, alpha=0.3)

ax.set(yscale="log")
ax.set(ylim=[1e-6, 1])
ax.set_yticklabels([])

ax.set_ylabel("")
ax.set_xlabel(r"\# parameter vectors sent and received")
ax.get_legend().remove()
# ax.set_title(r"$\sigma^2=0.1$ (w/ stochastic noise)")
ax.set(xlim=[0,80000])

plt.tight_layout()

#%%
f.savefig("../writing/neurips21/figures/social_graph_spanning_tree.pdf", bbox_inches="tight")
plt.close(f)

# %% Show the network
# import networkx as nx
# g = world.to_networkx()
# t = trees[1].to_networkx()
# # %%

# nx.draw(
#     g, 
#     pos=nx.kamada_kawai_layout(t), 
#     edge_color=["black" if e in t.edges else [0,0,0,0.2] for e in g.edges],
#     node_size=10,
#     width=[1 if e in t.edges else 0.5 for e in g.edges],
#     node_color="black"
# )
# plt.tight_layout()
# plt.savefig("../writing/neurips21/figures/social_graph_spanning_tree_illustration.pdf", bbox_inches="tight")
# # %%
