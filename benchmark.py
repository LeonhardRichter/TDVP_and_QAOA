from qaoa_and_tdvp import (
    QAOAResult,
    tdvp_optimize_qaoa,
    scipy_optimize,
)

from typing import List, Dict, Any
import pandas as pd
import pickle

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

    def run(
        self,
        qaoa,
        delta_0,
        p: int = None,
        tdvp_stepsize: float = None,
        tdvp_grad_tol: float = None,
        tdvp_lineq_solver: str = None,
    ) -> None:
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
