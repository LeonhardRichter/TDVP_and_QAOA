from qaoa_and_tdvp import (
    QAOAResult,
    tdvp_optimize_qaoa,
    scipy_optimize,
)

from typing import List, Dict, Any, Union
import pandas as pd
import pickle
import numpy as np
import networkx as nx


def get_rn_qubo(size: int, num: int = 1) -> np.matrix:
    qubos = list()
    for _ in range(num):
        rn = np.random.uniform(-1, 1, size=(size, size))
        qubos.append((rn + rn.T) / 2)

    if num == 1:
        return qubos[0]
    if num != 1:
        return qubos


def get_connected_rn_graph(
    number_of_nodes: int,
    p: float,
    number_of_graphs: int = 1,
) -> Union[nx.Graph,list[nx.Graph]]:
    assert 0 <= p <= 1, "p must be between 0 and 1"
    selected_graphs = []
    while len(selected_graphs) < number_of_graphs:
        graph = nx.fast_gnp_random_graph(number_of_nodes, p)
        if nx.is_connected(graph):
            selected_graphs.append(graph)
    if number_of_graphs == 1:
        return selected_graphs[0]
    if number_of_graphs != 1:
        return selected_graphs


def select_if_connected(graph: nx.Graph, number_of_nodes: int) -> nx.Graph | None:
    if graph.number_of_nodes() == number_of_nodes and nx.is_connected(graph):
        return graph


def get_all_connected(n: int) -> list[nx.Graph]:
    assert n <= 7, "only up to 7 nodes supported"
    return list(filter(lambda g: select_if_connected(g, n), nx.graph_atlas_g()))


class Benchmark:
    def __init__(self) -> None:
        self._results = list()

    # results
    @property
    def results(self) -> List[Dict[str, Any]]:
        """The list of results from the benchmark"""
        return self._results

    @results.setter
    def results(self, value: List[Dict[str, Any]]) -> None:
        self._results = value

    def test_run(
        self,
        qaoa,
        delta_0,
        p: int = None,
        tdvp_stepsize: float = None,
        tdvp_grad_tol: float = None,
        tdvp_lineq_solver: str = None,
    ) -> None:
        tdvp_res = QAOAResult()
        scipy_res = QAOAResult()

        return {
            "tdvp": tdvp_res,
            "scipy": scipy_res,
            "p": p,
            "delta_0": delta_0,
            "tdvp_stepsize": tdvp_stepsize,
            "tdvp_grad_tol": tdvp_grad_tol,
            "tdvp_lineq_solver": tdvp_lineq_solver,
        }

    def run(
        self,
        qaoa,
        delta_0,
        p: int = None,
        tdvp_stepsize: float = None,
        tdvp_grad_tol: float = None,
        tdvp_lineq_solver: str = None,
    ) -> None:
        if p is not None:
            qaoa.p = p
        tdvp_res = tdvp_optimize_qaoa(
            qaoa=qaoa,
            delta_0=delta_0,
            Delta=tdvp_stepsize,
            grad_tol=tdvp_grad_tol,
            int_mode=tdvp_lineq_solver,
        )
        scipy_res = scipy_optimize(delta_0=delta_0, qaoa=qaoa)
        self.results.append(
            {
                "tdvp": tdvp_res,
                "scipy": scipy_res,
                "p": p,
                "delta_0": delta_0,
                "tdvp_stepsize": tdvp_stepsize,
                "tdvp_grad_tol": tdvp_grad_tol,
                "tdvp_lineq_solver": tdvp_lineq_solver,
            }
        )

    def save(self, filename: str) -> None:
        df = pd.DataFrame(self.results)
        df.to_csv(f"./benchmarks/{filename}.csv")
        pickle.dump(self, open(f"./benchmarks/{filename}.p", "wb"))
