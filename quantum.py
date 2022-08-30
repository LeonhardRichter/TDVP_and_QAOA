from qutip import *
from qutip.qip.operations import expand_operator, rz

import numpy as np


from numpy.typing import ArrayLike, NDArray

from itertools import permutations


def sz(n: int, i: int):
    return expand_operator(sigmaz(), n, i, [2 for _ in range(n)])


def sx(n: int, i: int):
    return expand_operator(sigmax(), n, i, [2 for _ in range(n)])


def rzz(arg_value):
    return tensor(rz(arg_value), rz(-arg_value))


minus = (basis(2, 0) - basis(2, 1)).unit()


def q_j(qubo: ArrayLike) -> NDArray:
    qj = np.array([np.sum(row) - row[j] for j, row in enumerate(qubo)])
    return qj


def H_from_qubo(qubo: ArrayLike, constant: float = None) -> Qobj:
    n = qubo.shape[0]
    qj = q_j(qubo)
    if constant == None:
        qconstant = Qobj(
            np.full((2**n, 2**n), 0),
            dims=[[2 for _ in range(n)], [2 for _ in range(n)]],
        )
    else:
        qconstant = constant * qeye([2 for _ in range(n)])
    H = (
        sum([(qubo[i][i] + qj[i]) * sz(n, i) for i in range(n)])
        + sum([qubo[j][k] * sz(n, j) * sz(n, k) for j, k in permutations(range(n), 2)])
        + qconstant
    )
    return H
