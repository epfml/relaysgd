import itertools
import math
from typing import Any, Dict, Iterable

import networkx
import torch


class Topology:
    num_workers: int

    def __init__(self, num_workers):
        self.num_workers = num_workers

    def neighbors(self, worker: int) -> Iterable[int]:
        raise NotImplementedError()

    def degree(self, worker: int) -> int:
        return len(self.neighbors(worker))

    @property
    def workers(self) -> Iterable[int]:
        return range(self.num_workers)

    @property
    def max_degree(self) -> int:
        return max([self.degree(w) for w in self.workers])

    def gossip_matrix(self, weight=None) -> torch.Tensor:
        m = torch.zeros([self.num_workers, self.num_workers])
        for worker in self.workers:
            for neighbor in self.neighbors(worker):
                max_degree = max(self.degree(worker), self.degree(neighbor))
                m[worker, neighbor] = 1 / (max_degree + 1) if weight is None else weight
            # self weight
            m[worker, worker] = 1 - m[worker, :].sum()

        return m
    
    def to_networkx(self) -> networkx.Graph:
        g = networkx.Graph()
        g.add_nodes_from(range(self.num_workers))
        for worker in range(self.num_workers):
            g.add_edges_from([(worker, neighbor) for neighbor in self.neighbors(worker)])
        return g

    @property
    def max_delay(self):
        g = self.to_networkx()
        distances = dict(networkx.all_pairs_shortest_path_length(g))
        return max(distances[i][j] for i in g.nodes for j in g.nodes)


def configure_topology(config: Dict[str, Any]) -> Topology:
    if config["topology"] == "ring":
        return RingTopology(num_workers=config["distributed_world_size"])
    elif config["topology"] == "chain":
        return ChainTopology(num_workers=config["distributed_world_size"])
    elif config["topology"] == "3-tree":
        return TreeTopology(num_workers=config["distributed_world_size"], max_degree=3)
    elif config["topology"] == "binary-tree":
        return BinaryTreeTopology(num_workers=config["distributed_world_size"])
    elif config["topology"] == "double-binary-trees":
        return [
            BinaryTreeTopology(num_workers=config["distributed_world_size"]),
            BinaryTreeTopology(num_workers=config["distributed_world_size"], reverse=True)
        ]
    elif config["topology"] == "fully-connected":
        return FullyConnectedTopology(num_workers=config["distributed_world_size"])
    elif config["topology"] == "social-network":
        topology = SocialNetworkTopology()
        assert len(topology) == config["distributed_world_size"]
        return topology
    elif config["topology"] == "social-network-tree":
        topology = SocialNetworkTreeTopology(config["network_root_node"])
        assert len(topology) == config["distributed_world_size"]
        return topology
    else:
        raise ValueError("Unknown topology {}".format(config["topology"]))


class FullyConnectedTopology(Topology):
    def neighbors(self, worker):
        i = worker
        n = self.num_workers
        return [j for j in range(n) if j != i]


class StarTopology(Topology):
    def neighbors(self, worker):
        i = worker
        if i == 0:
            n = self.num_workers
            return [j for j in range(n) if j != i]
        else:
            return [0]


class ChainTopology(Topology):
    def neighbors(self, worker):
        if worker < 1:
            return [1]
        elif worker >= self.num_workers - 1:
            return [worker - 1]
        else:
            return [worker - 1, worker + 1]


class RingTopology(Topology):
    def neighbors(self, worker):
        i = worker
        n = self.num_workers
        return [(i - 1) % n, (i + 1) % n]


class HyperCubeTopology(Topology):
    def neighbors(self, worker):
        i = worker
        n = self.num_workers

        d = int(math.log2(n))
        assert 2 ** d == n

        return [i ^ (2 ** j) for j in range(0, d)]


class TreeTopology(Topology):
    """A tree that divides nodes such that nodes have the same degree if they are not (close to) leaves"""

    num_workers: int
    max_degree: int

    def __init__(self, num_workers, max_degree):
        super().__init__(num_workers=num_workers)
        self._max_degree = max_degree

    def max_workers_up_to_depth(self, layer_number: int) -> int:
        d = self._max_degree
        n = layer_number
        return int(1 + d * ((d - 1) ** n - 1) / (d - 2))

    def depth_of_worker(self, worker_number: int) -> int:
        # TODO: optimize / give direct formula
        depth = 0
        while True:
            if self.max_workers_up_to_depth(depth) > worker_number:
                return depth
            depth += 1

    def parent(self, worker_number: int) -> int:
        depth = self.depth_of_worker(worker_number)
        if depth == 0:
            return None
        index_within_layer = worker_number - self.max_workers_up_to_depth(depth - 1)
        if depth == 1:
            parent_within_layer = index_within_layer // (self._max_degree)
        else:
            parent_within_layer = index_within_layer // (self._max_degree - 1)
        return parent_within_layer + self.max_workers_up_to_depth(depth - 2)

    def children(self, worker_number: int) -> Iterable[int]:
        if worker_number == 0:
            children = [1 + x for x in range(self._max_degree)]
        else:
            depth = self.depth_of_worker(worker_number)
            start_idx_my_depth = self.max_workers_up_to_depth(depth - 1)
            start_idx_next_depth = self.max_workers_up_to_depth(depth)
            i = worker_number - start_idx_my_depth
            d = self._max_degree
            children = [start_idx_next_depth + (d - 1) * i + x for x in range(d - 1)]
        return [c for c in children if c < self.num_workers]

    def neighbors(self, worker: int) -> Iterable[int]:
        if worker == 0:
            return self.children(worker)
        else:
            return [self.parent(worker)] + self.children(worker)


class NetworkxTopology(Topology):
    def __init__(self, nx_graph):
        super().__init__(num_workers=len(nx_graph.nodes))
        self.graph = networkx.relabel.convert_node_labels_to_integers(nx_graph)

    def neighbors(self, worker: int) -> Iterable[int]:
        return list(self.graph.neighbors(worker))


class SocialNetworkTopology(NetworkxTopology):
    def __init__(self):
        nx_graph = networkx.davis_southern_women_graph()
        super().__init__(nx_graph)


class SocialNetworkTreeTopology(NetworkxTopology):
    def __init__(self, root_node):
        g = networkx.davis_southern_women_graph()
        nx_graph = self.best_spanning_tree_with_root(g, root_node)
        super().__init__(nx_graph)

    @staticmethod
    def best_spanning_tree_with_root(nx_graph, root_node):
        g = networkx.relabel.convert_node_labels_to_integers(nx_graph)
        edges = set()
        for n in g.nodes:
            path = sorted(networkx.all_shortest_paths(g, root_node, n))[0]
            for i, j in zip(path[:-1], path[1:]):
                edges.add((i, j))

        gg = networkx.Graph()
        for n in g.nodes:
            gg.add_node(n)
        gg.add_edges_from(edges)
        assert networkx.is_tree(gg)

        return gg


class BinaryTreeTopology(Topology):
    def __init__(self, num_workers, reverse=False):
        super().__init__(num_workers=num_workers)
        self.reverse = reverse

    def neighbors(self, worker):
        if self.num_workers < 2:
            return []
        elif worker >= self.num_workers or worker < 0:
            raise ValueError(f"worker number {worker} is out of range [0, {self.num_workers})")
        elif worker == 0 and not self.reverse:
            return [1]
        elif worker == self.num_workers - 1 and self.reverse:
            return [self.num_workers - 2]
        elif not self.reverse:
            parent = worker // 2
            children = [worker * 2, worker * 2 + 1]
            children = [c for c in children if c < self.num_workers]
            return [parent, *children]
        elif self.reverse:
            worker = self.num_workers - 1 - worker
            parent = worker // 2
            children = [worker * 2, worker * 2 + 1]
            children = [
                self.num_workers - 1 - c for c in children if c < self.num_workers
            ]
            parent = self.num_workers - 1 - parent
            return [parent, *children]
