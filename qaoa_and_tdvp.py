#%%
from time import time as ttime
from abc import ABC, abstractmethod
from itertools import combinations, combinations_with_replacement, product, permutations
from typing import Callable, Any, Iterable

from numba import njit

import numpy as np
from numpy.typing import NDArray, ArrayLike

import scipy as sp
from scipy import linalg
from scipy import integrate
from scipy.optimize import minimize

from qutip import *
from qutip.parallel import parallel_map
from qutip.qip.operations import expand_operator, rz
from qutip.qip.circuit import QubitCircuit, Gate

from quantum import *

#%%
# Define general expressions and objects


#%%
# Define qaoa class
class QAOAResult:
    def __init__(self) -> None:
        pass

    @property
    def optimal_parameters(self) -> tuple[float]:
        """The optimal_parameters property."""
        return self._optimal_parameters

    @optimal_parameters.setter
    def optimal_parameters(self, value: tuple[float]):
        self._optimal_parameters = value

    @property
    def optimal_state(self) -> Qobj():
        """The optimal_state property."""
        return self._optimal_state

    @optimal_state.setter
    def optimal_state(self, value: Qobj()):
        self._optimal_state = value

    @property
    def duration(self) -> float:
        """The duration property."""
        return self._duration

    @duration.setter
    def duration(self, value: float):
        self._duration = value

    @property
    def num_steps(self) -> int:
        """The num_steps property."""
        try:
            return self._num_steps
        except AttributeError:
            return None

    @num_steps.setter
    def num_steps(self, value: int):
        self._num_steps = value

    @property
    def num_fun_calls(self) -> int:
        """The num_fun_calls property."""
        return self._num_fun_calls

    @num_fun_calls.setter
    def num_fun_calls(self, value: int):
        self._num_fun_calls = value

    @property
    def optimal_fun_value(self) -> float:
        """The optimal_fun_value property."""
        return self._optimal_fun_value

    @optimal_fun_value.setter
    def optimal_fun_value(self, value: float):
        self._optimal_fun_value = value

    @property
    def parameter_path(self) -> list[tuple[float]]:
        """The parameter_path property."""
        return self._parameter_path

    @parameter_path.setter
    def parameter_path(self, value: list[tuple[float]]):
        self._parameter_path = value

    @property
    def success(self) -> bool:
        """The sucess property."""
        return self._sucess

    @success.setter
    def success(self, value: bool):
        self._sucess = value

    @property
    def optimizer_name(self) -> str:
        """The optimizer_name property."""
        return self._optimizer_name

    @optimizer_name.setter
    def optimizer_name(self, value: str):
        self._optimizer_name = value

    # message
    @property
    def message(self) -> str:
        """The message property."""
        return self._message

    @message.setter
    def message(self, value: str) -> None:
        self._message = value

    def __str__(self):
        return f"""
        {self.optimizer_name} terminated with {'no' if not self.success else''} sucess with message
        \"{self.message}\"

        optimal parameters:     {self.optimal_parameters}
        optimal value:          {self.optimal_fun_value}
        number of fun calls:    {self.num_fun_calls}
        number of steps:        {self.num_steps}
        """


class Optimizer(ABC):
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        return ""

    @abstractmethod
    def optimize(
        self,
        fun: Callable[[tuple[float]], float],
        delta_0: tuple[float],
        max_iter: int = 1000,
    ) -> QAOAResult:
        pass


#%%
class ScipyOptimizer(Optimizer):
    def __init__(self, gram=None, grad=None) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "ScipyOptimizer"

    def optimize(
        self,
        fun: Callable[[tuple[float]], float],
        delta_0: tuple[float],
        max_iter: int = 1000,
    ) -> QAOAResult:
        opt_result = QAOAResult()
        t_0 = ttime()
        min_result = minimize(
            fun, x0=delta_0, method="COBYLA", options={"maxiter": max_iter}
        )
        dt = ttime() - t_0
        opt_result.duration = dt
        opt_result.success = min_result.success
        opt_result.optimal_parameters = min_result.x
        opt_result.message = min_result.message
        opt_result.optimal_fun_value = min_result.fun
        opt_result.num_fun_calls = min_result.nfev
        try:
            opt_result.num_steps = min_result.nit
        except AttributeError:
            pass
        opt_result.optimizer_name = self.name

        return opt_result


#%%


class QAOA:
    def __init__(
        self,
        hamiltonian: Qobj = None,
        hamiltonian_ground: Qobj = None,
        qubo: ArrayLike = None,
        p: int = 1,
        optimizer: Optimizer = None,
    ) -> None:

        self._H = hamiltonian
        self.H_ground = hamiltonian_ground
        self.qubo = qubo
        self.p = p

        if self.qubo is not None:
            self._n = qubo.shape[0]
            self._qj = q_j(qubo)
        if self.H is not None:
            self._n = len(self.H.dims[0])

        self._optimizer = optimizer

        self.mixer_ground = tensor([minus for _ in range(self.n)])

        # set the H qubo-circuit depending on inputmode
        if self.qubo is None:
            self.qcH = self._qcHhamiltonian
        else:
            self.qcH = self._qcHqubo

    # qj
    @property
    def qj(self) -> NDArray:
        """sum over one jth row of self.qubo except the diagonal value"""
        return self._qj

    @qj.setter
    def qj(self, value: NDArray):
        self._qj = value

    # hamiltonian_ground
    @property
    def H_ground(self) -> Qobj:
        """The H_ground property."""
        return self._H_ground

    @H_ground.setter
    def H_ground(self, value: Qobj):
        self._H_ground = value

    # mixer ground
    @property
    def mixer_ground(self) -> Qobj:
        """The mxier_ground property."""
        return self._mixer_ground

    @mixer_ground.setter
    def mixer_ground(self, value: Qobj):
        self._mixer_ground = value

    # H
    @property
    def H(self) -> Qobj:
        """The H property."""
        if self._H == None:
            if self.qubo is not None:
                self._H = H_from_qubo(self.qubo)
        return self._H

    @H.setter
    def H(self, value: Qobj):
        self._H = value

    # n
    @property
    def n(self) -> int:
        """The n property."""
        if self._n == None:
            self._n = len(self.H.dims[0])
        return self._n

    @n.setter
    def n(self, value: int):
        self._n = value

    # optimizer
    @property
    def optimizer(self) -> Optimizer:
        """The optimizer property."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: Optimizer):
        self._optimizer = value

    ##############################################################################################

    def _qcHqubo(self, gamma: float) -> QubitCircuit:
        qc = QubitCircuit(self.n)
        qc.user_gates = {"RZZ", rzz}
        for j in range(self.n):
            qc.add_gate(
                "RZ",
                targets=j,
                arg_value=2 * gamma * (self.qubo[j][j] + self.qj[j]),
                arg_label=f"2*{round(gamma,2)}*(Q_{j}{j}+Q_{j})",
            )
        for j, k in combinations(range(self.n), 2):
            qc.add_gate(
                "RZZ",
                targets=[j, k],
                arg_value=2 * gamma * self.qubo[j][k],
                arg_label=f"2*{round(gamma,2)}*Q_{j}{k}",
            )
        return qc

    # define the qaoa gates as QubitCircuits
    def _qcHhamiltonian(self, gamma: float) -> QubitCircuit:
        def H_exp(arg_value) -> Qobj:
            return (-1j * arg_value * self.H).expm()

        qc = QubitCircuit(self.n)
        qc.user_gates = {"H_exp": H_exp}
        qc.add_gate("H_exp", arg_value=gamma, arg_label=f"{round(gamma,2)}")
        return qc

    def _qcB(self, beta: float) -> QubitCircuit:
        qc = QubitCircuit(self.n)
        qc.add_1q_gate("RX", arg_value=2 * beta, arg_label=f"2*{round(beta,2)}")
        return qc

    # define the whole qaoa circuit
    def circuit(self, delta: tuple[float]) -> QubitCircuit:
        assert len(delta) == 2 * self.p
        p = self.p
        n = self.n
        betas = delta[:p]
        gammas = delta[p : 2 * p]

        # define mixer circuit
        qc = QubitCircuit(n)

        for i in range(p):
            qc.add_circuit(self.qcH(gammas[i]))
            qc.add_circuit(self._qcB(betas[i]))

        return qc

    def circuit_i(
        self, delta: tuple, opers: list[Gate], i: int, tilde: bool = False
    ) -> QubitCircuit:
        """Compute the qaoa circuit with some gates inserted at a certain position.

        Args:
            delta (tuple): point ot look at.
            opers (list[Gate]): gates to insert
            i (int): qaoa layer for the gates to be inserted at
            tilde (bool, optional): Type of the insertion. If set to True, the gates are inserted inbetween H and B.
                                    If set to False, the gates are inserted after B. Defaults to False.

        Returns:
            QubitCircuit: the quantum circuit for this qaoa insertion.
        """
        qubo = self.qubo
        H = self.H
        n = self.n
        p = self.p
        assert len(delta) == 2 * p

        qc = QubitCircuit(n)
        if tilde:
            for layer in range(p):
                qc.add_circuit(self.qcH(delta[i + p]))
                if layer == i:
                    for oper in opers:
                        qc.add_gate(oper)
                qc.add_circuit(self._qcB(delta[i]))
        else:
            for layer in range(p):
                qc.add_circuit(self.qcH(delta[i + p]))
                qc.add_circuit(self._qcB(delta[i]))
                if layer == i:
                    for oper in opers:
                        qc.add_gate(oper)
        return qc

    def state(self, delta: tuple[float]) -> Qobj:
        return self.circuit((delta)).run(self.mixer_ground)

    def expectation(self, delta: tuple[float]) -> float:
        assert len(delta) == 2 * self.p
        return expect(self.H, self.state(delta))

    def solve(self, delta_0: tuple[float], max_iter=1000) -> QAOAResult:
        result = self.optimizer.optimize(
            fun=self.expectation, delta_0=delta_0, max_iter=max_iter
        )
        result.optimal_state = self.state(result.optimal_parameters)

        return result

    def _Adouble(self, left: tuple, right: tuple, delta) -> np.complex_:
        """compute one summand of G_ij, i.e. compute the overlap of two states left and right.
        Each of those is a QAOA state with some operators inserted at a certain position

        Args:
            left  (tuple): (H1:[Gate],i:int,tilde:bool) H1 is the list of operators to insert, i is the qaoa layer to insert at, tilde determines whether to insert inbetween or after the two qaoa gates
            right (tuple): (H2:[Gate],j:int,tilde:bool) see left
        """
        left_state = self.circuit_i(delta, *left).run(self.mixer_ground)
        right_state = self.circuit_i(delta, *right).run(self.mixer_ground)
        return (left_state.dag() * right_state)[0, 0]

    def _Asingle(self, left: tuple, right: tuple, delta) -> np.complex_:
        """compute one summand of G_ij, i.e

        Args:
            left  (tuple): (H1:[Gate],i:int,tilde:bool)
            right (tuple): (H2:[Gate],j:int,tilde:bool)
        """
        m_delta = tuple(-t for t in delta)
        qc = QubitCircuit(self.n)
        qc.add_circuit(self.circuit_i(delta, *right))
        # leftqc = U_i(m_delta,*left)
        # leftqc.add_1q_gate("SNOT") # add hadamards on every qubit to change basis (minus state now is )
        qc.add_circuit(
            self.circuit_i(m_delta, *left).reverse_circuit()
        )  # negative of the delta parameters gives in this case the adjoint gates
        return (self.mixer_ground.dag() * qc.run(self.mixer_ground))[0, 0]

    def _Gij(
        self, ij: tuple[int], delta: tuple[float], grammode: str = "double"
    ) -> np.complex_:
        i, j = ij
        n = self.n
        qubo = self.qubo
        qj = self.qj
        if grammode == "double":
            A = self._Adouble
        elif grammode == "single":
            A = self._Asingle
        if i <= p - 1 and j <= p - 1:
            element = sum(
                [
                    A(
                        ([Gate("X", [l])], i % p, False),  # left side
                        ([Gate("X", [k])], j % p, False),  # right side
                        delta,
                    )
                    for k, l in product(range(n), repeat=2)
                ]
            )

        if i <= p - 1 < j:

            element = sum(
                [
                    (qubo[l][l] + qj[l])
                    * A(
                        ([Gate("X", [k])], i % p, False),
                        (
                            [Gate("Z", [l])],
                            j % p,
                            True,
                        ),  # for the case that gamma was derivated: put a Z inbetween -> tilde=True
                        delta,
                    )
                    for k, l in product(range(n), repeat=2)
                ]
            ) + sum(
                [
                    2
                    * qubo[l][m]
                    * A(
                        ([Gate("X", [k])], i % p, False),
                        ([Gate("Z", [l]), Gate("Z", [m])], j % p, True),
                        delta,
                    )  # 2* is due to qubo being symmetric and not upper triangular
                    for k, l, m in product(range(n), repeat=3)
                    if l < m
                ]
            )
            # for saving computations, the gram matrix must be hermetian anyways
        if j <= p - 1 < i:
            element = sum(
                [
                    (qubo[l][l] + qj[l])
                    * A(
                        ([Gate("Z", [l])], i % p, True),
                        ([Gate("X", [k])], j % p, False),
                        delta,
                    )
                    for k, l in product(range(n), repeat=2)
                ]
            ) + sum(
                [
                    2
                    * qubo[k][l]
                    * A(
                        ([Gate("Z", [k]), Gate("Z", [l])], i % p, True),
                        ([Gate("X", [m])], j % p, False),
                        delta,
                    )  # 2* is due to qubo being symmetric and not upper triangular AND that it does not matter where which Z lands
                    for k, l, m in product(range(n), repeat=3)
                    if k < l
                ]
            )

        if i > p - 1 and j > p - 1:
            element = (
                sum(
                    [
                        qubo[k, k]
                        * (qubo[l, l] + qj[l])
                        * A(
                            ([Gate("Z", [k])], i % p, True),
                            ([Gate("Z", [l])], j % p, True),
                            delta,
                        )
                        for k, l in product(range(n), repeat=2)
                    ]
                )
                + sum(
                    [
                        2
                        * qubo[k, l]
                        * (qubo[m, m] + qj[l])
                        * A(
                            ([Gate("Z", [k]), Gate("Z", l)], i % p, True),
                            ([Gate("Z", [m])], j % p, True),
                            delta,
                        )
                        for k, l, m in product(range(n), repeat=3)
                        if k < l
                    ]
                )
                + sum(
                    [
                        2
                        * (qubo[k, k] + qj[k])
                        * qubo[l, m]
                        * A(
                            ([Gate("Z", [k])], i % p, True),
                            ([Gate("Z", [l]), Gate("Z", [m])], j % p, True),
                            delta,
                        )
                        for k, l, m in product(range(n), repeat=3)
                        if l < m
                    ]
                )
                + sum(
                    [
                        2
                        * qubo[k, l]
                        * 2
                        * qubo[m, n]
                        * A(
                            ([Gate("Z", [k]), Gate("Z", [l])], i % p, True),
                            ([Gate("Z", [m]), Gate("Z", [n])], j % p, True),
                            delta,
                        )
                        for k, l, m, n in product(range(n), repeat=4)
                        if k < l and m < n
                    ]
                )
            )
        return element

    def gram(self, delta: tuple[float], gram_mode: str = "double") -> NDArray:
        """Evaluate the gram matrix of an qubo-qaoa state on a quantum circuit.

        Args:
            delta (tuple): the parameter point to evaluate the matrix at.
            H (Qobj): the hamiltonian of the qaoa.
            qubo (NDArray): the qubo matrix in symmetric form .
            gram_mode (str, optional): Mode for the evaluation. Must be either 'double' or 'single'. If set to 'double', each side of the inner product is computed
            on its own, and then the product is eavlauated. If set to 'single', the gates of both factors are merged into one circuit. Defaults to 'double'.

        Returns:
            NDArray: The gram matrix of the qaoa with given parameters.
        """
        # #######legacy code########
        # n = self.n
        # p = self.p
        # U_i = self.circuit_i
        # qubo = self.qubo
        # # two methods: one for computing both sides sepeartely (2 circuits) and for one longer circuit
        # if gram_mode == "double":

        #     def A(left: tuple, right: tuple, delta) -> np.complex_:
        #         """compute one summand of G_ij, i.e. compute the overlap of two states left and right.
        #         Each of those is a QAOA state with some operators inserted at a certain position

        #         Args:
        #             left  (tuple): (H1:[Gate],i:int,tilde:bool) H1 is the list of operators to insert, i is the qaoa layer to insert at, tilde determines whether to insert inbetween or after the two qaoa gates
        #             right (tuple): (H2:[Gate],j:int,tilde:bool) see left
        #         """
        #         left_state = U_i(delta, *left).run(self.mixer_ground)
        #         right_state = U_i(delta, *right).run(self.mixer_ground)
        #         return (left_state.dag() * right_state)[0, 0]

        # elif gram_mode == "single":

        #     def A(left: tuple, right: tuple, delta) -> np.complex_:
        #         """compute one summand of G_ij, i.e

        #         Args:
        #             left  (tuple): (H1:[Gate],i:int,tilde:bool)
        #             right (tuple): (H2:[Gate],j:int,tilde:bool)
        #         """
        #         m_delta = tuple(-t for t in delta)
        #         qc = QubitCircuit(n)
        #         qc.add_circuit(U_i(delta, *right))
        #         # leftqc = U_i(m_delta,*left)
        #         # leftqc.add_1q_gate("SNOT") # add hadamards on every qubit to change basis (minus state now is )
        #         qc.add_circuit(
        #             U_i(m_delta, *left).reverse_circuit()
        #         )  # negative of the delta parameters gives in this case the adjoint gates
        #         return (self.mixer_ground.dag() * qc.run(self.mixer_ground))[0, 0]

        # # initialize the matrix
        # G = np.zeros((2 * p, 2 * p), dtype=np.complex128)
        ###### current parallized version#######
        return np.matrix(
            parallel_map(
                task=self._Gij,
                values=list(product(range(2 * p), repeat=2)),
                task_args=(delta, gram_mode),
            )
        ).reshape(2 * self.p, 2 * self.p)

    def _grad_element(self, i, delta: tuple[float], dummy):
        if i <= self.p - 1:
            circ = self.circuit_i(
                delta, [Gate("X", [k]) for k in range(self.n)], i % self.p, tilde=False
            )
            return (circ.run(self.mixer_ground).dag() * self.H * self.state(delta))[
                0, 0
            ]

        if i > self.p - 1:
            if self.qubo is None:

                def H_gate() -> Qobj:
                    assert (
                        self.H.isunitary
                    ), "Hamiltonian is not unitary. Can't use it for gradient evaluation."
                    return self.H

                circ = self.circuit_i(delta, [H_gate], i % self.p, tilde=True)

                return (circ.run(self.mixer_ground).dag() * self.H * self.state(delta))[
                    0, 0
                ]

            # if qubo was given, use it for implementing H by Z gates
            if self.qubo is not None:
                state = sum(
                    [
                        (self.qubo[k, k] + self.qj[k])
                        * (
                            self.circuit_i(
                                delta, [Gate("Z", [k])], i % self.p, tilde=True
                            ).run(self.mixer_ground)
                        )
                        for k in range(self.n)
                    ]
                ) + sum(
                    [
                        # factor of two because qubo is symmetric and we only add each gate combination once
                        2
                        * self.qubo[k, l]
                        * (
                            self.circuit_i(
                                delta,
                                [Gate("Z", [k]), Gate("Z", [l])],
                                i % self.p,
                                tilde=True,
                            ).run(self.mixer_ground)
                        )
                        for k, l in combinations(range(self.n), 2)
                    ]
                )
                return (state.dag() * self.H * self.state(delta))[0, 0]

    def grad(self, delta: tuple[float]):
        return np.matrix(
            parallel_map(self._grad_element, range(len(delta)), task_args=(delta, 0))
        )

    def grad_legacy(self, delta: tuple[float]) -> NDArray:
        states = []  # list for saving the states
        # first half -> beta derivatives
        for i in range(self.p):
            circ = self.circuit_i(
                delta, [Gate("X", [k]) for k in range(self.n)], i, tilde=False
            )
            states.append(
                circ.run(self.mixer_ground).dag() * self.H * self.state(delta)
            )
        # second half -> gamma derivatives
        for i in range(self.p):
            # if no qubo known just apply the hamiltonian as gate
            if self.qubo is None:

                def H_gate() -> Qobj:
                    assert (
                        self.H.isunitary
                    ), "Hamiltonian is not unitary. Can't use it for gradient evaluation."
                    return self.H

                circ = self.circuit_i(delta, [H_gate], i, tilde=True)
                states.append(
                    circ.run(self.mixer_ground).dag() * self.H * self.state(delta)
                )
            # if qubo was given, use it for implementing H by Z gates
            if self.qubo is not None:
                state = sum(
                    [
                        self.qubo[k, k]
                        * (
                            self.circuit_i(delta, [Gate("Z", [k])], i, tilde=True).run(
                                self.mixer_ground
                            )
                        )
                        for k in range(self.n)
                    ]
                ) + sum(
                    [
                        # factor of two because qubo is symmetric and we only add each gate combination once
                        2
                        * self.qubo[k, l]
                        * (
                            self.circuit_i(
                                delta, [Gate("Z", [k]), Gate("Z", [l])], i, tilde=True
                            ).run(self.mixer_ground)
                        )
                        for k, l in combinations(range(self.n), 2)
                    ]
                )
                states.append(state.dag() * self.H * self.state(delta))
        return np.matrix(state).T


# %%
class tdvp_optimizer(Optimizer):
    def __init__(
        self,
        state_param: Callable[[tuple[float]], Qobj],
        hamiltonian: Qobj,
        gram: Callable[[tuple[float]], NDArray] = None,
        grad: Callable[[tuple[float]], NDArray] = None,
        Delta: float = 10 ** (-3),
    ) -> None:
        super().__init__()
        self._hamiltonian = hamiltonian
        self._state_param = state_param
        self._Delta = Delta
        if gram is None:
            self._gram = self.gen_gram
        else:
            self._gram = gram
        if grad is None:
            self._grad = self.gen_grad
        else:
            self._grad = grad

    # Delta
    @property
    def Delta(self) -> float:
        """The time stepssize."""
        return self._Delta

    @Delta.setter
    def Delta(self, value: float):
        self._Delta = value

    # hamiltonian
    @property
    def hamiltonian(self) -> Qobj:
        """The hamiltonian property."""
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, value: Qobj):
        self._hamiltonian = value

    # state_param
    @property
    def state_param(self) -> Callable[[tuple[float]], Qobj]:
        """The state_param property."""
        return self._state_param

    @state_param.setter
    def state_param(self, value: Callable[[tuple[float]], Qobj]) -> None:
        self._state_param = value

    # gram
    @property
    def gram(self) -> Callable[[tuple[float]], NDArray]:
        """The gram matrix to be used."""
        return self._gram

    @gram.setter
    def gram(self, value: Callable[[tuple[float]], NDArray]) -> None:
        self._gram = value

    # grad
    @property
    def grad(self) -> Callable[[tuple[float]], NDArray]:
        """The gradient to be used."""
        return self._grad

    @grad.setter
    def grad(self, value: Callable[[tuple[float]], NDArray]) -> None:
        self._grad = value

    def name(self) -> str:
        return "tdvp_optimizer"

    def finitediff(
        self, func: Callable[[Iterable[float]], Any], epsilon: float = 1e-10
    ) -> Callable[[Iterable[float]], list[Any]]:
        def dfunc(params):
            params = tuple(params)
            difference = list()
            for i in range(2):
                p_params = list(params)
                p_params[i] += epsilon
                difference.append((func(p_params) - func(params)) / epsilon)
            return difference

        return dfunc

    def gen_grad(
        self,
        pars: tuple[float],
    ) -> np.matrix:
        dpsi = self.finitediff(self.state_param)
        p = int(len(pars))
        out = np.matrix(
            [
                (dpsi(pars)[k].dag() * self.hamiltonian * self.state_param(pars))[0, 0]
                for k in range(p)
            ]
        )
        return out

    def gen_gram(self, pars: tuple[float]) -> np.matrix:
        dpsi = self.finitediff(self.state_param)
        p = int(len(pars))
        return np.matrix(
            [
                [((dpsi(pars)[j]).dag() * dpsi(pars)[k])[0, 0] for k in range(p)]
                for j in range(p)
            ]
        )  # order of j,k must be correct -> j should be rows, k should be columns

    def flow(self, delta_0: tuple[float], Delta=None):
        if Delta == None:
            Delta = self.Delta

        def RHS(t, x):
            """right hand side of linear equation system of tdvp. In the right format for the scipy solvers.

            Args:
                t (float): time variable
                x (tuple[float]): parameter input

            Returns:
                NDArray: the matrix defining the RHS of the equation
            """
            imag_gram = linalg.inv(np.imag(self.gram(x)))
            real_grad = np.real(self.grad(x))
            return np.array(-imag_gram * real_grad.T).flatten()

        result = integrate.solve_ivp(
            fun=RHS,
            t_span=(0, Delta),
            t_eval=[0, Delta],
            y0=delta_0,
            method="RK45",
        )
        return result

    def optimize(
        self,
        delta_0: tuple[float],
        max_iter: int = 1000,
        threshhold: float = 10**-10,
        Delta=None,
    ) -> QAOAResult:
        if Delta is None:
            Delta = self.Delta
        delta = delta_0
        for _ in range(max_iter):
            flow_result = self.flow(delta)
            delta = tuple(flow_result.y[_, -1] for _ in range(len(flow_result.y)))
            print(delta)
            if linalg.norm(self.grad(delta)) < threshhold:
                break
        opt_result = QAOAResult()
        opt_result.duration = 0
        opt_result.success = 0
        opt_result.optimal_parameters = delta
        opt_result.message = ""
        opt_result.optimal_fun_value = (
            self.state_param(delta).dag() * self.hamiltonian * self.state_param(delta)
        )[0, 0]
        opt_result.num_fun_calls = 0
        return opt_result


#%%
p = 1
qubo = np.array([[1, 3], [3, 4]])
qaoa = QAOA(qubo=qubo, p=p)
delta = tuple(0.1 for _ in range(2 * p))
qaoa.optimizer = ScipyOptimizer()

# %%

tdvp = tdvp_optimizer(
    state_param=qaoa.state,
    hamiltonian=qaoa.H,
    # gram=qaoa.gram,
    # grad=qaoa.grad,
    Delta=0.05,
)

# %%
tdvp.optimize(delta_0=delta, max_iter=100)

# %%
