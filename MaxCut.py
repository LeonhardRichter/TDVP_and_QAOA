# ==========================================
# Author: Leonhard Felix Richter
# Date:   25 Jan 2023
# ==========================================
"""This module defines the MaxCut class allowing to generate the qubo matrix and
the corresponding hamiltonian for a given graph.
"""
#%%
import networkx as nx
from numpy.typing import NDArray
import numpy as np
from qutip import Qobj
from quantum import H_from_qubo

#%%


class MaxCut:
    def __init__(self, graph: nx.Graph, draw: bool = True) -> None:
        self.graph = graph
        for (i, j) in self.graph.edges:
            if "weight" not in self.graph[i][j].keys():
                self.graph[i][j]["weight"] = 1
        self._qubo = self._get_qubo(self.graph)
        self.H = self._get_hamiltonian()
        if draw:
            nx.draw(self.graph, with_labels=True)

    # qubo
    @property
    def qubo(self) -> NDArray:
        """The qubo matrix for the mac-cut instance corresponding to self.graph"""
        return self._qubo

    @qubo.setter
    def qubo(self, value: NDArray) -> None:
        self._qubo = value

    def _get_qubo(self, graph: nx.Graph | None = None) -> NDArray:
        if graph is None:
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

    def _get_hamiltonian(self) -> Qobj:
        self.H = H_from_qubo(self.qubo)
        return self.H


# %%
