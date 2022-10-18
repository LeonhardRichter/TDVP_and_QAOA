from benchmark import Benchmark
from quantum import H_from_qubo, Qobj
from MaxCut import MaxCut

from qutip.parallel import parallel_map, serial_map

from qaoa_and_tdvp import (
    QAOA,
    QAOAResult,
    qaoa_tdvp_rhs,
    tdvp_optimize_qaoa,
    scipy_optimize,
)

import pickle

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Union, Optional, Callable, Any
import networkx as nx
import matplotlib.pyplot as plt
import plotly as py

import pandas as pd


n = 4
with open(f"./instances/n{n}_instances.p", "rb") as f:
    instances = pickle.load(f)
delta = tuple(1 for _ in range(2 * n))
bench = Benchmark()


Delta = 0.1
min_p = 1
max_p = 8


for n in range(7, 8):
    with open(f"./instances/n{n}_instances.p", "rb") as f:
        instances = pickle.load(f)
    delta = tuple(1 for _ in range(2 * n))
    bench = Benchmark()

    for num, i in enumerate(instances):
        print(f"instance {num}/{len(instances)}")
        for p in range(min_p, max_p + 1):
            print(f"\t p = {p}/{max_p}")
            bench.run(
                qaoa=QAOA(i.qubo, p=p),
                delta_0=tuple(1 for _ in range(2 * p)),
                tdvp_stepsize=1,
                tdvp_grad_tol=1e-3,
                p=p,
                tdvp_lineq_solver="RK45",
            )
            bench.save(f"RK45_n{n}_p-1-10_Delta-1_benchmarks")
