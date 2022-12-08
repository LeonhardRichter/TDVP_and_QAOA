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
from itertools import product
import pickle
from scipy.linalg import LinAlgError
import numpy as np
import networkx as nx
from MaxCut import MaxCut


def get_rn_qubo(size: int, num: int = 1) -> np.matrix | list[np.matrix]:
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
    else:
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
    else:
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
        instance: MaxCut,
        delta_0,
        p: int | None = None,
        tdvp_range: float | None = None,
        tollarance: float | None = None,
        tdvp_lineq_solver: str | None = None,
    ) -> dict:
        tdvp_res = QAOAResult()
        scipy_res = QAOAResult()

        return {
            "instance": instance,
            "tdvp_res": tdvp_res,
            "scipy_res": scipy_res,
            "p": p,
            "delta_0": delta_0,
            "tollarance": tollarance,
            "tdvp_lineq_solver": tdvp_lineq_solver,
        }

    def run(
        self,
        instance: MaxCut,
        delta_0,
        p: int = 1,
        tdvp_range: float = 1.0,
        tollarance: float = 1e-2,
        tdvp_lineq_solver: str = "RK45",
        max_iter: int = 200,
        record_path: bool = False,
    ) -> None:
        qaoa = QAOA(qubo=instance.qubo, p=p)
        if p is not None:
            qaoa.p = p

        try:
            scipy_res = scipy_optimize(
                delta_0=delta_0, qaoa=qaoa, record_path=record_path, tol=tollarance
            )
        except LinAlgError:
            scipy_res = QAOAResult()
            scipy_res.success = False
            scipy_res.message = "LinAlgError"
            scipy_res.qaoa = qaoa

        try:
            tdvp_res = tdvp_optimize_qaoa(
                qaoa=qaoa,
                delta_0=delta_0,
                Delta=tdvp_range,
                grad_tol=tollarance,
                int_mode=tdvp_lineq_solver,
                max_iter=max_iter,
            )
        except LinAlgError:
            tdvp_res = QAOAResult()
            tdvp_res.success = False
            tdvp_res.message = "LinAlgError"
            tdvp_res.qaoa = qaoa

        self.results.append(
            {
                "tdvp": tdvp_res,
                "scipy": scipy_res,
                "p": p,
                "delta_0": delta_0,
                "tollarance": tollarance,
                "tdvp_lineq_solver": tdvp_lineq_solver,
            }
        )

    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self, f)


def load_instances(n: int, p_min: int = 1, p_max: int = 5) -> pd.DataFrame:
    with open(f"./instances/n{n}_instances.p", "rb") as f:
        instances = pickle.load(f)

    instances = dict(enumerate(instances))

    arrays = [
        list(range(p_min, p_max + 1)),
        list(instances.keys()),
    ]
    tuples = product(*arrays)
    index = pd.MultiIndex.from_tuples(tuples, names=["p", "i"])
    df = pd.DataFrame(
        index=index, columns=["instance", "tdvp", "scipy", "gradient_descent"]
    )
    # df['instances'] = df.apply(lambda x: 0, axis=0)
    for (p, i) in df.index:
        df["instance"][(p, i)] = instances[i]
    return df


class bench_result(dict):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

    def __repr__(self) -> str:
        return f'{self.get("algorithm", np.NaN)}, C={self.get("value", np.NaN)}, delta={self.get("delta", np.NaN)}'


def bench_recursive(
    input: MaxCut | pd.DataFrame | pd.Series,
    p: int = 1,
    optimizers: dict[str, bool] = {
        "tdvp": True,
        "scipy": True,
        "gradient_descent": False,
    },
    tollarance: float = 1e-2,
    tdvp_range: float = 1.0,
    max_iter: int = 200,
    max_steps: int = 1000,
    auto_save: bool = False,
    path: str | None = None,
    print_msg: bool = True,
) -> pd.DataFrame | pd.Series:
    """Benchmark function for one maxcut instance and one p value

    Args:
        instance (MaxCut | pd.Series): The instance to benchmark or a pandas series having a field "instance" with the instances and a field "p" with the p value
        p (int): the qaoa depth
        optimizers (dict[str,bool]): dictionary of optimizers to use.
        Only keys "tdvp", "scipy" and "gradient_descent" are recognized

    Returns:
        dict: The result of the benchmark in a form that can be directly converted to a pandas dataframe
    """

    if isinstance(input, MaxCut):
        if print_msg:
            print(
                f"Running benchmark on instance with {input.graph.number_of_nodes()} with optimizers {(key for key,value in optimizers.items() if value)}"
            )
        qaoa = QAOA(input.qubo, p=p)
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
                max_iter=max_iter,
                max_steps=max_steps,
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

        out = {
            algo: bench_result(
                {
                    "instance": input,
                    # "qaoa": qaoa,   # produces unnecessary large files
                    "p": p,
                    "n": qaoa.n,
                    "delta_0": delta_0,
                    "tollarance": tollarance,
                    "algorithm": algo,
                    "res": res,
                    "delta": res.parameters,
                    "value": res.value,
                    "path": res.parameter_path,
                    "steps": res.num_steps,
                    "num_fun_calls": res.num_fun_calls,
                    "real duration": res.duration,
                    "message": res.message,
                }
            )
            for algo, res in results.items()
        }
        for k in out.keys():
            out[k].__repr__ = lambda: f"{k}: {out[k]['value']}"

        # return the results
        return pd.DataFrame(data=out, columns=["tdvp", "scipy", "gradient_descent"])

    elif isinstance(input, pd.Series):
        out = bench_recursive(
            input["instance"],
            p=input.name[0],  # type: ignore
            optimizers=optimizers,
            tollarance=tollarance,
            max_iter=max_iter,
            max_steps=max_steps,
            auto_save=auto_save,
            path=path,
            print_msg=False,
        )
        input["tdvp"], input["scipy"], input["gradient_descent"] = (
            out["tdvp"],
            out["scipy"],
            out["gradient_descent"],
        )
        return input
    elif isinstance(input, pd.DataFrame):
        if print_msg:
            print(
                f"Running benchmark on {len(input)} instances with optimizers {(key for key,value in optimizers.items() if value)}"
            )
        return input.apply(
            lambda x: bench_recursive(
                x,
                optimizers=optimizers,
                tollarance=tollarance,
                max_iter=max_iter,
                max_steps=max_steps,
                auto_save=auto_save,
                path=path,
                print_msg=False,
            ),  # type: ignore
            axis=1,
        )
    else:
        raise TypeError(
            f"instance must be either a MaxCut instance or a pandas series of MaxCut instances or a pandas Series but is {type(input)}"
        )


def bench_instance(
    input: MaxCut,
    p: int = 1,
    optimizers: dict[str, bool] = {
        "tdvp": True,
        "scipy": True,
        "gradient_descent": False,
    },
    tollarance: float = 1e-2,
    max_iter: int = 1,
    max_steps: int = 100,
    tdvp_range: float = 1,
    auto_save: bool = False,
    path: str | None = None,
    print_msg: bool = True,
) -> pd.DataFrame:
    if print_msg:
        print(
            f"Running benchmark on instance with {input.graph.number_of_nodes()} with optimizers {tuple(key for key,value in optimizers.items() if value)}"
        )
    qaoa = QAOA(input.qubo, p=p)
    delta_0 = tuple(1 for _ in range(2 * p))
    results: dict[str, QAOAResult] = dict()

    # get the results
    if optimizers.get("tdvp", False):
        try:
            print("optimizing with tdvp")
            tdvp_res = tdvp_optimize_qaoa(
                qaoa=qaoa,
                delta_0=delta_0,
                Delta=100,
                grad_tol=tollarance,
                int_mode="RK45",
                max_iter=max_iter,
                max_steps=max_steps,
            )
        except LinAlgError:
            print(f"LinAlgError at p={p} and instance {input}")
            tdvp_res = QAOAResult()
            tdvp_res.success = False
            tdvp_res.message = "LinAlgError"
            tdvp_res.duration = 0
            tdvp_res.parameters = delta_0
            tdvp_res.num_fun_calls = 0
        except ValueError:
            print(f"ValueError at p={p} and instance {input}")
            tdvp_res = QAOAResult()
            tdvp_res.success = False
            tdvp_res.message = "ValueError"
            tdvp_res.duration = 0
            tdvp_res.parameters = delta_0
            tdvp_res.num_fun_calls = 0

        results["tdvp"] = tdvp_res

    if optimizers.get("scipy", False):
        try:
            print("optimizing with scipy")
            scipy_res = scipy_optimize(
                delta_0=delta_0, qaoa=qaoa, record_path=True, tol=tollarance
            )
        except LinAlgError:
            print(f"LinAlgError at p={p} and instance {input}")
            scipy_res = QAOAResult()
            scipy_res.success = False
            scipy_res.message = "LinAlgError"
            scipy_res.duration = 0
            scipy_res.parameters = delta_0
            scipy_res.num_fun_calls = 0
        except ValueError:
            print(f"ValueError at p={p} and instance {input}")
            tdvp_res = QAOAResult()
            tdvp_res.success = False
            tdvp_res.message = "ValueError"
            tdvp_res.duration = 0
            tdvp_res.parameters = delta_0
            tdvp_res.num_fun_calls = 0

        results["scipy"] = scipy_res

    if optimizers.get("gradient_descent", False):
        print("optimizing with gradient descent")
        gradient_descent_res = gradient_descent(
            delta_0=delta_0,
            qaoa=qaoa,
            tol=tollarance,
            max_iter=max_steps,
        )
        results["gradient_descent"] = gradient_descent_res

    out = {
        algo: {
            "instance": input,
            # "qaoa": qaoa,   # produces unnecessary large files
            "p": p,
            "n": qaoa.n,
            "delta_0": delta_0,
            "tollarance": tollarance,
            "algorithm": algo,
            "res": res,
            "delta": res.parameters,
            "value": res.value,
            "path": res.parameter_path,
            "steps": res.num_steps,
            "num_fun_calls": res.num_fun_calls,
            "real duration": res.duration,
            "message": res.message,
        }
        for algo, res in results.items()
    }

    # return the results
    return pd.DataFrame(data=out, columns=["tdvp", "scipy", "gradient_descent"])


def bench_series(
    input: pd.Series,
    p: int | None = None,
    optimizers: dict[str, bool] = {
        "tdvp": True,
        "scipy": True,
        "gradient_descent": False,
    },
    tollarance: float = 1e-2,
    max_iter: int = 1,
    max_steps: int = 100,
    tdvp_range: float = 1,
    auto_save: bool = False,
    path: str | None = None,
    print_msg: bool = True,
) -> pd.DataFrame | pd.Series:
    try:
        out = bench_instance(
            input["instance"],
            p=input.name[0],  # type: ignore
            optimizers=optimizers,
            tollarance=tollarance,
            max_iter=max_iter,
            max_steps=max_steps,
            tdvp_range=tdvp_range,
            auto_save=auto_save,
            path=path,
            print_msg=False,
        )
    except LinAlgError:
        out = input
        print(f"LinAlgError at {input.name}")
    except ValueError:
        out = input
        print(f"ValueError at {input.name}")

    input["tdvp"], input["scipy"], input["gradient_descent"] = (
        out["tdvp"],
        out["scipy"],
        out["gradient_descent"],
    )
    return input


def bench_frame(
    input: pd.DataFrame,
    p: int | None = None,
    optimizers: dict[str, bool] = {
        "tdvp": True,
        "scipy": True,
        "gradient_descent": False,
    },
    tollarance: float = 1e-2,
    max_iter: int = 1,
    max_steps: int = 100,
    tdvp_range: float = 1,
    auto_save: bool = False,
    path: str | None = None,
    print_msg: bool = True,
) -> pd.DataFrame:
    if print_msg:
        print(
            f"Running benchmark on {len(input)} instances with optimizers {tuple(key for key,value in optimizers.items() if value)}"
        )
    out = input.apply(
        lambda x: bench_series(
            x,
            optimizers=optimizers,
            tollarance=tollarance,
            max_iter=max_iter,
            max_steps=max_steps,
            tdvp_range=tdvp_range,
            auto_save=auto_save,
            path=path,
            print_msg=False,
        ),  # type: ignore
        axis=1,
    )
    # type: ignore
    if path is not None:
        with open(path, "wb") as f:
            pickle.dump(out, f)

    return out


def bench_looping(
    input: pd.DataFrame,
    p: int | None = None,
    optimizers: dict[str, bool] = {
        "tdvp": True,
        "scipy": True,
        "gradient_descent": False,
    },
    tollarance: float = 1e-2,
    max_iter: int = 1,
    max_steps: int = 1000,
    auto_save: bool = False,
    path: str | None = None,
    print_msg: bool = True,
) -> pd.DataFrame:
    df = input
    if print_msg:
        print(
            f"Running benchmark on {len(input)} instances with optimizers {tuple(key for key,value in optimizers.items() if value)}"
        )
    for (p, i) in input.index:
        try:
            df.loc[(p, i)] = (  # type: ignore
                bench_series(
                    df.loc[(p, i)],  # type: ignore
                    p=p,
                    optimizers=optimizers,
                    tollarance=tollarance,
                    max_iter=max_iter,
                    max_steps=max_steps,
                    auto_save=auto_save,
                    path=path,
                    print_msg=False,
                ),
            )
        except ValueError:
            print(f"Error at p={p}, i={i}")
            continue

    if path is not None:
        with open(path, "wb") as f:
            pickle.dump(df, f)

    return df
