from itertools import combinations, permutations
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import List
from qutip import sigmaz, sigmax, basis, tensor, qeye, Qobj
from qutip.qip.operations import expand_operator, rz


def H_exp(arg_value, H) -> Qobj:
    return (-1j * arg_value * H).expm()


# a 'cheat' gate just giving back the input. Used for the efficient simulation of the metric
def cheat_gate(H: Qobj) -> Qobj:
    return H


def sz(n: int, i: int) -> Qobj:
    return expand_operator(sigmaz(), n, i, [2 for _ in range(n)])


def sx(n: int, i: int) -> Qobj:
    return expand_operator(sigmax(), n, i, [2 for _ in range(n)])


def rzz_old(arg_value) -> Qobj:
    return tensor(rz(arg_value), rz(-arg_value))


def rzz(arg_val) -> Qobj:
    a = np.zeros((4, 4), dtype=np.complex128)
    np.fill_diagonal(
        a,
        [
            np.exp(-1j * arg_val / 2),
            np.exp(1j * arg_val / 2),
            np.exp(1j * arg_val / 2),
            np.exp(-1j * arg_val / 2),
        ],
    )
    return Qobj(a, dims=[[2, 2], [2, 2]])


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
        # + qconstant
    )
    return H

def groundspace(H:Qobj):
    eigenenergies, eigenstates = H.eigenstates(sort='low')
    groundenergy = eigenenergies[0]

    num_of_eigenstates = 0
    for e in eigenenergies:
        if e==groundenergy:
            num_of_eigenstates +=1
        if e != groundenergy:
            break
    return groundenergy, eigenstates[:num_of_eigenstates]

def groundspace_overlap(groundstates:List[Qobj], state:Qobj):
    return  sum(abs(g.overlap(state))**2 for g in groundstates)

def groundspace_sharpness(groundstates:List[Qobj], state:Qobj):
    P_0 = sum(g*g.dag() for g in groundstates)
    return math.sqrt(variance(P_0,state))
