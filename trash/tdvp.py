#%%
from time import process_time, time

from itertools import combinations, combinations_with_replacement
from typing import Callable, Iterable

import numpy as np
from numpy import pi
from numpy.typing import NDArray, ArrayLike, DTypeLike

import scipy as sp
from scipy import linalg
from scipy import integrate
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import plotly.express as px

from qutip import *
from qutip.qip.operations import *
from qutip.qip.circuit import QubitCircuit, Gate

# %%

#%%


def sz(n: int, i: int):
    return expand_operator(sigmaz(), n, i, [2 for _ in range(n)])


def sx(n: int, i: int):
    return expand_operator(sigmax(), n, i, [2 for _ in range(n)])


minus = (basis(2, 0) - basis(2, 1)).unit()


def rzz(arg_value):
    return tensor(rz(arg_value), rz(-arg_value))


class tdvp_result:
    def __init__(self) -> None:
        pass

    @property
    def num_calls(self):
        """The num_calls property."""
        return self._num_calls

    @num_calls.setter
    def num_calls(self, value):
        self._num_calls = value

    @property
    def opt_pars(self):
        """The opt_pars property."""
        return self._opt_pars

    @opt_pars.setter
    def opt_pars(self, value):
        self._opt_pars = value

    @property
    def opt_state(self):
        """The opt_state property."""
        return self._opt_state

    @opt_state.setter
    def opt_state(self, value):
        self._opt_state = value

    @property
    def num_steps(self):
        """The num_steps property."""
        return self._num_steps

    @num_steps.setter
    def num_steps(self, value):
        self._num_steps = value

    @property
    def opt_goal_val(self):
        """The opt_goal_val property."""
        return self._opt_goal_val

    @opt_goal_val.setter
    def opt_goal_val(self, value):
        self._opt_goal_val = value


class tdvp_optimizer:
    def __init__(
        self,
        psi: Callable[[tuple[float]], Qobj],
        H: Qobj,
        qubo: NDArray,
        gram_mode: str = "double",
    ) -> None:
        self._psi = psi
        self._H = H
        assert qubo.T.all() == qubo.all(), "Qubo matrix not symmetric"
        self._qubo = qubo
        self.n = len(H.dims[0])
        assert gram_mode in {"single", "double"}
        self._gram_mode = gram_mode

    @property
    def gram_mode(self):
        """The gram_mode property."""
        return self._gram_mode

    @gram_mode.setter
    def gram_mode(self, value):
        self._gram_mode = value

    @property
    def qubo(self):
        """The qubo property."""
        return self._qubo

    @qubo.setter
    def qubo(self, value):
        self._qubo = value

    @property
    def psi(self):
        """The psi property."""
        return self._psi

    @psi.setter
    def psi(self, value):
        self._psi = value

    @property
    def H(self):
        """The H property."""
        return self._H

    @H.setter
    def H(self, value):
        self._H = value

    def flow(self, delta, stepsize) -> tuple:
        """flow according to tdvp from pars for stepsize duration.

        Args:
            pars (tuple): starting point parameters
            stepsize (float): stepsize of the flow

        Returns:
            _type_: _description_
        """

        def finitediff(
            func: Callable[[tuple[float]], Qobj], epsilon: float = 1e-10
        ) -> Callable[[Iterable[float]], Qobj]:
            def dfunc(delta):
                difference = list()
                for i in range(2):
                    p_delta = list(delta)
                    p_delta[i] += epsilon
                    difference.append((func(p_delta) - func(delta)) / epsilon)
                return difference

            return dfunc

        def qaoa_grad(
            psi: Callable[[tuple[float]], Qobj], delta: Iterable[float], H: Qobj
        ) -> np.matrix:
            dpsi = finitediff(psi)
            grad = np.matrix(
                [(dpsi(delta)[k].dag() * H * psi(delta))[0, 0] for k in range(2)]
            ).T
            return grad

    def optimize(self, tol, stepsize) -> tdvp_result:
        """Run the optimization algorithm using the tdvp flow.

        Args:
            tol (float): _description_
            stepsize (float): _description_

        Returns:
            tdvp_result: _description_
        """
        pass

    def get_qaoa_gram(self, delta: tuple) -> NDArray:
        """evaluate the gram matrix on a quantum circuit.

        Args:
            delta (tuple): tuple of qaoa parameters. In the first half are betas, in the second one gammas

        Returns:
            NDArray: Gram-matrix
        """
        p = int(len(delta) / 2)

        def U_i(delta: tuple, opers: list[Gate], i: int, tilde: bool = False):
            if self.qubo is not None:
                # define what to apply to the circuit in each H_p turn
                def qcH(gamma: float) -> QubitCircuit:
                    qc = QubitCircuit(self.n)
                    qc.user_gates = {"RZZ", rzz}
                    for j in range(self.n):
                        qc.add_gate(
                            "RZ",
                            targets=j,
                            arg_value=2 * gamma * self.qubo[j][j],
                            arg_label=f"2*{round(gamma,2)}*{self.qubo[j][j]}",
                        )
                    for j, k in combinations(range(self.n), 2):
                        qc.add_gate(
                            "RZZ",
                            targets=[j, k],
                            arg_value=2 * gamma * self.qubo[j][k],
                            arg_label=f"2*{round(gamma,2)}*{self.qubo[j][k]}",
                        )
                    return qc

            else:
                assert self.H.isherm

                def qcH(gamma: float) -> QubitCircuit:
                    def H_exp(arg_value):
                        return (-1j * arg_value * self.H).expm()

                    qc = QubitCircuit(self.n)
                    qc.user_gates = {"H_exp": H_exp}
                    qc.add_gate("H_exp", arg_value=gamma, arg_label=f"{round(gamma,2)}")
                    return qc

            def qcB(beta: float) -> QubitCircuit:
                qc = QubitCircuit(self.n)
                qc.add_1q_gate("RX", arg_value=2 * beta)
                return qc

            qc = QubitCircuit(self.n)
            if tilde:
                for layer in range(p):
                    qc.add_circuit(qcH(delta[i + p]))
                    if layer == i:
                        for oper in opers:
                            qc.add_gate(oper)
                    qc.add_circuit(qcB(delta[i]))
            else:
                for layer in range(p):
                    qc.add_circuit(qcH(delta[i + p]))
                    qc.add_circuit(qcB(delta[i]))
                    if layer == i:
                        for oper in opers:
                            qc.add_gate(oper)
            return qc

        # two methods: one for computing both sides sepeartely (2 circuits) and for one longer circuit

        if self.gram_mode == "single":

            def A(left: tuple, right: tuple, delta):
                """compute one summand of G_ij

                Args:
                    left  (tuple): (H1:[Gate],i:int,tilde:bool)
                    right (tuple): (H2:[Gate],j:int,tilde:bool)
                """
                right_state = U_i(delta,*right).run(tensor([minus for _ in range(self.n)]))
                left_state  = U_i(delta,*left ).run(tensor([minus for _ in range(self.n)]))
                return (left_state.dag()*right_state)[0,0]
    
        elif self.gram_mode == 'double':
            def A(left:tuple, right:tuple, delta):
                """compute one summand of G_ij

                Args:
                    left  (tuple): (H1:[Gate],i:int,tilde:bool)
                    right (tuple): (H2:[Gate],j:int,tilde:bool)
                """
                m_delta = tuple(-t for t in delta)
                qc = QubitCircuit(self.n)
                qc.add_circuit(U_i(delta, *right))
                # leftqc = U_i(m_delta,*left)
                # leftqc.add_1q_gate("SNOT") # add hadamards on every qubit to change basis (minus state now is )
                qc.add_circuit(
                    U_i(m_delta, *left).reverse_circuit()
                )  # negative of the delta parameters gives in this case the adjoint gates
                overlap = tensor([minus for _ in range(self.n)]).dag() * qc.run(
                    tensor([minus for _ in range(self.n)])
                )
                return overlap[0, 0]

        G = np.zeros((p, p), dtype=np.complex128)
        for i, j in combinations_with_replacement(range(p), 2):
            if i <= p and j <= p:
                G[i, j] = sum(
                    [
                        A(
                            ([Gate("X", [k])], i, False),
                            ([Gate("X", [l])], j, False),
                            delta,
                        )
                        for k, l in combinations_with_replacement(range(self.n), 2)
                    ]
                )

            if i <= p and j > p:
                G[i, j] = sum(
                    [
                        self.qubo[l][l]
                        * A(
                            ([Gate("X", [k])], i, False),
                            ([Gate("Z", [l])], j, True),
                            delta,
                        )
                        for k, l in combinations_with_replacement(range(self.n), 2)
                    ]
                ) + sum(
                    [
                        2
                        * self.qubo[l][m]
                        * A(
                            ([Gate("X", [k])], i, False),
                            ([Gate("Z", [l]), Gate("Z", [m])], j, True),
                            delta,
                        )  # 2* is due to qubo being symmetric and not upper triangular
                        for k, l, m in combinations_with_replacement(range(self.n), 3)
                        if l < m
                    ]
                )

            if i > p and j <= p:
                G[i, j] = sum(
                    [
                        self.qubo[k][k]
                        * A(
                            ([Gate("Z", [k])], i, True),
                            ([Gate("X", [l])], j, False),
                            delta,
                        )
                        for k, l in combinations_with_replacement(range(self.n), 2)
                    ]
                ) + sum(
                    [
                        2
                        * self.qubo[k][l]
                        * A(
                            ([Gate("Z", [k]), Gate("Z", [l])], i, True),
                            ([Gate("X", [m])], j, False),
                            delta,
                        )
                        for k, l, m in combinations_with_replacement(range(self.n), 3)
                        if k < l
                    ]
                )

            if i > p and j > p:
                G[i, j] = (
                    sum(
                        [
                            self.qubo[k, k]
                            * self.qubo[l, l]
                            * A(
                                ([Gate("Z", [k])], i, True),
                                ([Gate("Z", [l])], j, True),
                                delta,
                            )
                            for k, l in combinations_with_replacement(range(self.n), 2)
                        ]
                    )
                    + sum(
                        [
                            2
                            * self.qubo[k, l]
                            * self.qubo[m, m]
                            * A(
                                ([Gate("Z", [k]), Gate("Z", l)], i, True),
                                ([Gate("Z", [m])], j, True),
                                delta,
                            )
                            for k, l, m in combinations_with_replacement(
                                range(self.n), 3
                            )
                            if k < l
                        ]
                    )
                    + sum(
                        [
                            self.qubo[k, k]
                            * 2
                            * self.qubo[l, m]
                            * A(
                                ([Gate("Z", [k])], i, True),
                                ([Gate("Z", [l]), Gate("Z", [m])], j, True),
                                delta,
                            )
                            for k, l, m in combinations_with_replacement(
                                range(self.n), 3
                            )
                        ]
                    )
                    + sum(
                        [
                            2
                            * self.qubo[k, l]
                            * 2
                            * self.qubo[m, n]
                            * A(
                                ([Gate("Z", [k]), Gate("Z", [l])], i, True),
                                ([Gate("Z", [m]), Gate("Z", [n])], j, True),
                                delta,
                            )
                            for k, l, m, n in combinations_with_replacement(
                                range(self.n), 4
                            )
                            if k < l and m < n
                        ]
                    )
                )
        return np.matrix(G)


# %%
def psi(pars:tuple[float])->Qobj:
    return tensor([np.cos(pars[0]/2)*basis(2,0)+np.exp(1j*pars[1])*np.sin(pars[0]/2)*basis(2,1) for _ in range(2)])
            
tdvp_ = qaoa_tdvp(psi,tensor(sigmaz(),sigmaz()),np.array([[1,2],[2,1]]),gram_mode='double')
t_0 = time()
tdvp_.get_qaoa_gram((1, 1, 1, 1))
print(time() - t_0)

# %%
