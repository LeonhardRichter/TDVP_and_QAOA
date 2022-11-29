from qaoa_and_tdvp import (
    QAOAResult,
    tdvp_optimize_qaoa,
    scipy_optimize,
    gradient_descent,
    QAOA,
)
from math import prod
from typing import List, Dict, Any, Union
import pandas as pd
import swifter
import pickle
from scipy.linalg import LinAlgError
import numpy as np
import networkx as nx
from MaxCut import MaxCut
import os


def get_rn_qubo(size: int, num: int = 1) -> np.matrix:
    qubos = list()

    while len(qubos) < num:
        rn = np.random.uniform(-1, 1, size=(size, size))
        qubo = np.matrix(rn + rn.T) / 2
        # check if qubo is already in list

        if len(qubos) == 0:
            qubos.append(qubo)
            continue

        if not prod([np.all(np.isclose(qubo, a)) for a in qubos]):
            qubos.append(qubo)
            continue

    qubos = list(qubos)

    if num == 1:
        return qubos[0]
    if num != 1:
        return qubos


def get_connected_rn_graph(
    number_of_nodes: int,
    p: float,
    number_of_graphs: int = 1,
) -> Union[nx.Graph, list[nx.Graph]]:
    assert 0 <= p <= 1, "p must be between 0 and 1"
    selected_graphs = set()

    while len(selected_graphs) < number_of_graphs:
        graph = nx.fast_gnp_random_graph(number_of_nodes, p)
        if nx.is_connected(graph):
            selected_graphs.add(graph)

    selected_graphs = list(selected_graphs)

    if number_of_graphs == 1:
        return selected_graphs[0]
    if number_of_graphs != 1:
        return selected_graphs


def select_if_connected(graph: nx.Graph, number_of_nodes: int) -> Union[nx.Graph, None]:
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
        max_steps: int = 200,
        record_path: bool = False,
    ) -> None:
        if p is not None:
            qaoa.p = p

        try:
            scipy_res = scipy_optimize(
                delta_0=delta_0, qaoa=qaoa, record_path=record_path
            )
        except LinAlgError:
            scipy_res = QAOAResult()
            scipy_res.success = False
            scipy_res.message = "LinAlgError"
            scipy_res.prob = 0
            scipy_res.qaoa = qaoa

        try:
            tdvp_res = tdvp_optimize_qaoa(
                qaoa=qaoa,
                delta_0=delta_0,
                Delta=tdvp_stepsize,
                grad_tol=tdvp_grad_tol,
                int_mode=tdvp_lineq_solver,
                max_iter=max_steps,
            )
        except LinAlgError:
            tdvp_res = QAOAResult()
            tdvp_res.success = False
            tdvp_res.message = "LinAlgError"
            tdvp_res.prob = 0
            tdvp_res.qaoa = qaoa

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


def benchmark_pandas(
    instance: MaxCut | pd.DataFrame,
    p: int,
    optimizers: dict[str, bool] = {
        "tdvp": True,
        "scipy": True,
        "gradient_descent": True,
    },
    tollarance: float = 1e-2,
    auto_save: bool = False,
    path: str = None,
) -> pd.DataFrame:
    """Benchmark function for one maxcut instance and one p value

    Args:
        instance (MaxCut | pd.Series): The instance to benchmark or a pandas series having a field "instance" with the instances and a field "p" with the p value
        p (int): the qaoa depth
        optimizers (dict[str,bool]): dictionary of optimizers to use.
        Only keys "tdvp", "scipy" and "gradient_descent" are recognized

    Returns:
        dict: The result of the benchmark in a form that can be directly converted to a pandas dataframe
    """
    if isinstance(instance, MaxCut):
        print(
            f"Running benchmark on instance with {instance.graph.number_of_nodes()} with optimizers {(key for key,value in optimizers.items() if value)}"
        )
        qaoa = QAOA(instance.qubo, p=p)
        delta_0 = tuple(1 for _ in range(2 * p))
        results: dict[str, QAOAResult] = dict()

        # get the results
        if optimizers.get("tdvp", False):
            print("optimizing with tdvp")
            tdvp_res = tdvp_optimize_qaoa(
                qaoa=qaoa,
                delta_0=delta_0,
                Delta=100,
                grad_tol=tollarance,
                int_mode="RK45",
                max_iter=100,
            )
            results["tdvp"] = tdvp_res

        if optimizers.get("scipy", False):
            print("optimizing with scipy")
            scipy_res = scipy_optimize(
                delta_0=delta_0, qaoa=qaoa, record_path=True, tol=tollarance
            )
            results["scipy"] = scipy_res

        if optimizers.get("gradient_descent", False):
            print("optimizing with gradient descent")
            gradient_descent_res = gradient_descent(
                delta_0=delta_0, qaoa=qaoa, tol=tollarance
            )
            results["gradient_descent"] = gradient_descent_res

        if auto_save and path is not None:
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    results = pickle.load(f).extend(results)
            with open(path, "wb") as f:
                pickle.dump(results, f)

        # return the results
        return pd.DataFrame(data=[
            {
                "instance": instance,
                "qaoa": qaoa,
                "p": p,
                "n": qaoa.n,
                "delta_0": delta_0,
                "tollarance": tollarance,
                "algorithm": algo,
                "res": res,
                "delta": res.parameters,
                "path": res.parameter_path,
                "steps": res.num_steps,
                "num_fun_calls": res.num_fun_calls,
                "real duration": res.duration,
                "message": res.message,
            }
            for algo,res in results.items()
        ])

    elif isinstance(instance, pd.DataFrame):
        df = pd.DataFrame()
        instance.swifter.apply(
            lambda x: df.append(
                benchmark_pandas(
                    x["instance"],
                    x["p"],
                    optimizers=optimizers,
                    tollarance=tollarance,
                    auto_save=auto_save,
                    path=path,
                )
            ),
            axis=1,
        )
        return df
    else:
        raise TypeError(
            "instance must be either a MaxCut instance or a pandas series of MaxCut instances"
        )
