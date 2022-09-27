from itertools import permutations
import numpy as np
from numpy.typing import ArrayLike, NDArray
from qutip import sigmaz, sigmax, basis, tensor, qeye, Qobj
from qutip.qip.operations import expand_operator, rz


def H_exp(arg_value, H) -> Qobj:
    return (-1j * arg_value * H).expm()


def sz(n: int, i: int) -> Qobj:
    return expand_operator(sigmaz(), n, i, [2 for _ in range(n)])


def sx(n: int, i: int) -> Qobj:
    return expand_operator(sigmax(), n, i, [2 for _ in range(n)])


def rzz(arg_value) -> Qobj:
    return tensor(rz(arg_value), rz(-arg_value))


minus = (basis(2, 0) - basis(2, 1)).unit()


def q_j(qubo: ArrayLike) -> NDArray:
    qj = np.array([np.sum(row) - row[j] for j, row in enumerate(qubo)])
    return qj


def H_from_qubo(qubo: ArrayLike, constant: float = None) -> Qobj:
    n = qubo.shape[0]
    qj = q_j(qubo)
    # handle constant term
    if constant == None:
        constant = 0
    qconstant = (
        constant + np.sum(np.diagonal(qubo)) + np.sum(qubo - np.diag(np.diagonal(qubo)))
    ) * qeye([2 for _ in range(n)])
    H = (
        (-1 / 2) * sum((qubo[i][i] + qj[i]) * sz(n, i) for i in range(n))
        + (1 / 4)
        * sum(qubo[j][k] * sz(n, j) * sz(n, k) for j, k in permutations(range(n), 2))
        + qconstant
    )
    return H
