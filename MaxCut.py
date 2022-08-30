#%%
import networkx as nx

import numpy as np
from numpy.typing import NDArray

from itertools import combinations

from quantum import *

#%%


class MaxCut:
    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph
        for (i, j) in self.graph.edges:
            if "weight" not in self.graph[i][j].keys():
                self.graph[i][j]["weight"] = 1

    # qubo
    @property
    def qubo(self) -> NDArray:
        """The qubo matrix for the mac-cut instance corresponding to self.graph"""
        try:
            return self._qubo
        except AttributeError:
            self._qubo = self.get_qubo()
            return self._qubo

    @qubo.setter
    def qubo(self, value: NDArray):
        self._qubo = value

    def get_qubo(self, graph: nx.Graph = None) -> NDArray:
        if graph == None:
            graph = self.graph
        n = graph.number_of_nodes()
        Q = np.zeros((n, n), dtype=np.float64)
        for (i, j) in graph.edges:
            Q[i, j] = Q[j, i] = graph[i][j]["weight"]

        np.fill_diagonal(
            Q,
            [
                np.sum(
                    np.fromiter(
                        (graph[i][j]["weight"] for j in graph.neighbors(i)),
                        dtype=np.float64,
                    )
                )
                for i in graph.nodes
            ],
        )
        return Q

    def get_hamiltonian(self) -> None:
        self.H = H_from_qubo(self.qubo)
        return self.H


# %%
