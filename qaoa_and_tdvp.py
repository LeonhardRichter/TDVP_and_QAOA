# vscode-fold=2
# %%
from time import time as time, process_time
from itertools import combinations, product, combinations_with_replacement
from typing import Callable, Tuple, Iterable

# import scipy as sp
from scipy import linalg
from scipy import integrate
from scipy.optimize import minimize

import numpy as np

from qutip.parallel import parallel_map, serial_map
from qutip import expect

# from qutip.qip.operations import expand_operator, rz
from qutip.qip.circuit import QubitCircuit, Gate
from qutip.ipynbtools import parallel_map as ipy_parallel_map

from quantum import *


# %%
# Define general expressions and objects


# %%
# Define qaoa class
def rzz(arg_value) -> Qobj:
    return tensor(rz(arg_value), rz(-arg_value))


# %%


class QAOA:
    def __init__(
        self,
        hamiltonian: Qobj = None,
        hamiltonian_ground: Qobj = None,
        qubo: NDArray = None,
        p: int = 1,
        grammode: str = "single",
    ) -> None:
        """

        :type qubo: NDArray
        """
        self._H = hamiltonian
        self.H_ground = hamiltonian_ground
        self.qubo = qubo
        self.p = p

        if self.qubo is not None:
            self._n = qubo.shape[0]
            self._qj = q_j(qubo)
        if self.H is not None:
            self._n = len(self.H.dims[0])

        self.mixer_ground = tensor([minus for _ in range(self.n)])

        # set the H qubo-circuit depending on inputmode
        if self.qubo is None:
            self._qcH = self._qcHhamiltonian
        else:
            self._qcH = self._qcHqubo

        match grammode:
            case "double":
                self._A = self._Adouble
            case "single":
                self._A = self._Asingle

    # qj
    @property
    def qj(self) -> NDArray:
        """sum over one jth row of self.qubo except the diagonal value"""
        return self._qj

    @qj.setter
    def qj(self, value: NDArray) -> None:
        self._qj = value

    # hamiltonian_ground
    @property
    def H_ground(self) -> Qobj:
        """The H_ground property."""
        return self._H_ground

    @H_ground.setter
    def H_ground(self, value: Qobj) -> None:
        self._H_ground = value

    # mixer ground
    @property
    def mixer_ground(self) -> Qobj:
        """The mxier_ground property."""
        return self._mixer_ground

    @mixer_ground.setter
    def mixer_ground(self, value: Qobj) -> None:
        self._mixer_ground = value

    # H
    @property
    def H(self) -> Qobj:
        """The H property."""
        if self._H is None and self.qubo is not None:
            self._H = H_from_qubo(self.qubo)
        return self._H

    @H.setter
    def H(self, value: Qobj) -> None:
        self._H = value

    # n
    @property
    def n(self) -> int:
        """The n property."""
        if self._n is None:
            self._n = len(self.H.dims[0])
        return self._n

    @n.setter
    def n(self, value: int) -> None:
        self._n = value

    ##############################################################################################

    def _qcHqubo(self, gamma: float) -> QubitCircuit:
        qc = QubitCircuit(self.n)
        qc.user_gates = {"RZZ": rzz}
        for j in range(self.n):
            qc.add_gate(
                "RZ",
                targets=[j],
                arg_value=2 * gamma * (self.qubo[j][j] + self.qj[j]),
                arg_label=f"2*{round(gamma, 2)}*(Q_{{{j}{j}}}+Q_{j})",
            )
        for j, k in combinations(range(self.n), 2):
            print("moin")
            qc.add_gate(
                "RZZ",
                targets=[j, k],
                arg_value=2 * gamma * self.qubo[j][k],
                arg_label=f"2*{round(gamma, 2)}*Q_{{{j}{k}}}",
            )
            # qc.add_gate(
            #     "RZZ",
            #     targets=[j, k],
            #     arg_value=2 * gamma * self.qubo[j, k],
            #     arg_label=f"2*{round(gamma, 2)}*Q_{{{j}{k}}}",
            # )
        return qc

    # define the qaoa gates as QubitCircuits
    def _qcHhamiltonian(self, gamma: float) -> QubitCircuit:
        def H_exp(arg_value) -> Qobj:
            return (-1j * arg_value * self.H).expm()

        qc = QubitCircuit(self.n)
        qc.user_gates = {"H_exp": H_exp}
        qc.add_gate("H_exp", arg_value=gamma, arg_label=f"{round(gamma, 2)}")
        return qc

    def _qcB(self, beta: float) -> QubitCircuit:
        qc = QubitCircuit(self.n)
        qc.add_1q_gate("RX", arg_value=2 * beta, arg_label=f"2*{round(beta, 2)}")
        return qc

    # define the whole qaoa circuit
    def circuit(self, delta: tuple[float]) -> QubitCircuit:
        assert len(delta) == 2 * self.p
        p = self.p
        n = self.n
        betas = delta[:p]
        gammas = delta[p : 2 * p]
        print(gammas)

        # define mixer circuit
        qc = QubitCircuit(n)

        for i in range(p):
            qc.add_circuit(self._qcH(gammas[i]))
            qc.add_circuit(self._qcB(betas[i]))

        return qc

    def circuitDiff_old(
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
        n = self.n
        p = self.p
        assert len(delta) == 2 * p

        qc = QubitCircuit(n)
        match tilde:
            case True:
                for layer in range(p):
                    qc.add_circuit(self._qcH(delta[i + p]))
                    if layer == i:
                        for oper in opers:
                            qc.add_gate(oper)
                    qc.add_circuit(self._qcB(delta[i]))
            case False:
                for layer in range(p):
                    qc.add_circuit(self._qcH(delta[i + p]))
                    qc.add_circuit(self._qcB(delta[i]))
                    if layer == i:
                        for oper in opers:
                            qc.add_gate(oper)
        return qc

    def circuitDiff(
        self,
        delta,
        gates: Iterable[Gate],
        position: int,
        pop_gates: Tuple[int, int] = None,
    ) -> QubitCircuit:
        """Compute the qaoa circuit with some gates inserted at a certain position, and some gates removed.

        Args:
            delta (_type_): point ot look at.
            gates (Iterable[Gate]): gates to insert
            position (int): gate index for the gates to be inserted at
            pop_gates (Tuple[int, int], optional): Gates to be removed. IMPORTANT: These gates may only be at higher indices than the inserted ones.
            Otherwise the indexing will go wrong. Defaults to None.

        Returns:
            QubitCircuit: the quantum circuit for this qaoa insertion.
        """
        qaoa = self.circuit(delta)  # the usual qaoa circuit
        if pop_gates is not None:
            qaoa.remove_gate_or_measurement(
                *pop_gates
            )  # possiblity to remove gates, do this before inserting new ones, otherwise the indexing will be wrong
        for gate in gates:  # for each given gate, insert it at the given position
            qaoa.add_gate(gate, index=[position])
        return qaoa

    def state(self, delta: tuple[float]) -> Qobj:
        return self.circuit(delta).run(self.mixer_ground)

    def expectation(self, delta: tuple[float]) -> float:
        assert len(delta) == 2 * self.p
        return expect(self.H, self.state(delta))

    def _Adouble(
        self, left: tuple, right: tuple, delta, pop_gates: Tuple[int, int] = None
    ) -> np.complex_:
        """compute one summand of G_ij, i.e. compute the overlap of two states left and right.
        Each of those is a QAOA state with some operators inserted at a certain position

        Args:
            left  (tuple): (left_gates:tuple[Gate],left_pos:int) left_gates is the list of operators to insert, left_pos is the gate index
                            to insert at
            right (tuple): (right_gates:tuple[Gate],right_pos:int) see left
        """
        left_state = self.circuitDiff(delta, *left, pop_gates=pop_gates).run(
            self.mixer_ground
        )
        right_state = self.circuitDiff(delta, *right, pop_gates=pop_gates).run(
            self.mixer_ground
        )
        return (left_state.dag() * right_state)[0, 0]

    def _Asingle(
        self, left: tuple, right: tuple, delta, pop_gates: Tuple[int, int] = None
    ) -> np.complex_:
        """compute one summand of G_ij, i.e. compute the combined circuit of left and right side and run it on input state.

        Args:
            left  (tuple): (left_gates:tuple[Gate],left_pos:int) positions are gate-wise not qaoa-layer wise
            right (tuple): (right_gates:tuple[Gate],right_pos:int)
        """
        m_delta = tuple(-t for t in delta)  # negative parameters
        qc = self.circuitDiff(delta, *right, pop_gates=pop_gates)
        qc.add_circuit(
            self.circuitDiff(m_delta, *left, pop_gates=pop_gates).reverse_circuit()
        )  # negative of the delta parameters gives in this case the adjoint gates
        return (self.mixer_ground.dag() * qc.run(self.mixer_ground))[0, 0]

    def _Gij(self, ij: tuple[int], delta: tuple[float]) -> np.complex_:
        """compute one summand of G_ij, i.e. decide in which part of the matrix the indices lie and compute the corresponding element.
            Assumes that i<=j, i.e. only the upper triangle of the matrix is computed directly. The lower triangle is computed by hermiticity.

        Args:
            ij (tuple[int]): indices of the matrix element
            delta (tuple[float]): parameters of the qaoa circuit

        Returns:
            np.complex_: the matrix element
        """
        i, j = ij
        n, p, qubo, qj, A = self.n, self.p, self.qubo, self.qj, self._A

        if i <= p - 1 and j <= p - 1:  # upper left corner of the matrix
            element = sum(
                A(
                    left=(
                        [Gate("X", [l])],
                        2 * i + 2,
                    ),  # positions are handpicked. Each Qaoa layer has 2 gates, the first one is H, the second one is B. For insertion after B, the position is 2*i+2, for insertion inbetween H and B, the position is 2*i+1
                    right=([Gate("X", [k])], 2 * j + 2),  # same as above
                    delta=delta,
                    # remove gates that will cancel each other out due to the adjoint circuit. This only works when i<=j! That is handled by self.gram only computing the upper triangle with this method
                    pop_gates=(
                        2 * j + 1,
                        2 * p,
                    ),
                )
                for k, l in product(
                    range(n), repeat=2
                )  # sum over all possible combinations of X-gates
            )

        elif i <= p - 1 < j:  # upper right corner of the matrix
            element = sum(
                (qubo[l][l] + qj[l])  # the linear qubo coefficients coming from Z
                * A(
                    left=([Gate("X", [k])], 2 * i + 2),  # insert X after B
                    right=([Gate("Z", [l])], 2 * j + 1),  # insert Z inbetween H and B
                    delta=delta,
                    pop_gates=(2 * j + 1, 2 * p),
                )
                for k, l in product(range(n), repeat=2)
            ) + sum(
                2  # 2* is due to qubo being symmetric and not upper triangular
                * qubo[l][m]  # the quadratic qubo coefficients coming from ZZ
                * A(
                    left=([Gate("X", [k])], 2 * i + 2),  # insert X after B
                    right=(
                        [Gate("Z", [l]), Gate("Z", [m])],
                        2 * j + 1,
                    ),  # insert two Z's inbetween H and B
                    delta=delta,
                    pop_gates=(
                        2 * j + 1,
                        2 * p,
                    ),  # remove gates that will cancel each other out due to the adjoint circuit.
                    # 2*j+1 is the positions of the inserted gate with highest index.
                    # As removement is done before insertion this is the first  gate that can be removed on both sides.
                )
                for k, (l, m) in product(
                    range(n), combinations(range(n), r=2)
                )  # sum over all possible combinations (l<m) of Z-gates, different orderings are counted twice
            )

        # the case j<=p-1<i is handled by hermiticity, as it lies in the lower triangle of the matrix

        elif i > p - 1 and j > p - 1:  # lower right corner of the matrix
            element = (
                sum(
                    (qubo[k, k] + qj[k])  # the linear qubo coefficients coming from Z_k
                    * (
                        qubo[l, l] + qj[l]
                    )  # the linear qubo coefficients coming from Z_l
                    * A(
                        left=([Gate("Z", [k])], 2 * i + 1),  # insert Z after H
                        right=([Gate("Z", [l])], 2 * j + 1),  # insert Z after H
                        delta=delta,
                        pop_gates=(2 * j + 1, 2 * p),
                    )
                    for k, l in product(
                        range(n), repeat=2
                    )  # sum over all possible combinations of Z-gates
                )
                + sum(
                    2  # 2* is due to qubo being symmetric and not upper triangular
                    * qubo[k, l]  # the quadratic qubo coefficients coming from Z_k Z_l
                    * (
                        qubo[m, m] + qj[l]
                    )  # the linear qubo coefficients coming from Z_m
                    * A(
                        left=(
                            [Gate("Z", [k]), Gate("Z", l)],
                            2 * i + 1,
                        ),  # insert two Z's after H
                        right=([Gate("Z", [m])], 2 * j + 1),  # insert Z after H
                        delta=delta,
                        pop_gates=(2 * j + 1, 2 * p),
                    )
                    for (k, l), m in product(
                        combinations(range(n), r=2), range(n)
                    )  # sum over all possible combinations (k<l) of Z-gates and (m) X Gates, different orderings are counted twice
                )
                + sum(
                    2  # 2* is due to qubo being symmetric and not upper triangular
                    * (
                        qubo[k, k] + qj[k]
                    )  # the linear qubo coefficients coming from Z_k
                    * qubo[l, m]  # the quadratic qubo coefficients coming from Z_l Z_m
                    * A(
                        left=([Gate("Z", [k])], 2 * i + 1),  # insert Z after H
                        right=(
                            [Gate("Z", [l]), Gate("Z", [m])],
                            2 * j + 1,
                        ),  # insert two Z's after H
                        delta=delta,
                        pop_gates=(2 * j + 1, 2 * p),
                    )
                    for k, (l, m) in product(
                        range(n), combinations(range(n), r=2)
                    )  # sum over all possible combinations (l<m) of Z-gates and (k) X Gates, different orderings are counted twice
                )
                + sum(
                    2  # 2* is due to qubo being symmetric and not upper triangular
                    * qubo[k, l]  # the quadratic qubo coefficients coming from Z_k Z_l
                    * 2  # 2* is due to qubo being symmetric and not upper triangular
                    * qubo[m, n]  # the quadratic qubo coefficients coming from Z_m Z_n
                    * A(
                        left=(
                            [Gate("Z", [k]), Gate("Z", [l])],
                            2 * i + 1,
                        ),  # insert two Z's after H
                        right=(
                            [Gate("Z", [m]), Gate("Z", [n])],
                            2 * j + 1,
                        ),  # insert two Z's after H
                        delta=delta,
                        pop_gates=(2 * j + 1, 2 * p),
                    )
                    for (k, l), (m, n) in product(
                        combinations(range(3), r=2), repeat=2
                    )  # sum over all possible combinations (k<l) of Z-gates and (m<n) of Z-gates, different orderings are counted twice
                )
            )
        return element

    def gram(self, delta: tuple[float], **kwargs) -> NDArray:
        """Evaluate the gram matrix of an qubo-qaoa state on a quantum circuit.

        Args:
            delta (tuple): parameters of the qaoa circuit
            gram_mode (str, optional): The mode in which the gram matrix is computed. Defaults to "single".
        Returns:
            NDArray: The gram matrix of the qaoa with given parameters.
        """
        gram = np.zeros((2 * self.p, 2 * self.p), dtype=np.complex128)
        # populate the upper triangular part and diagonal of the gram matrix using self._Gij
        # np.triu_indices returns the indices of the upper triangular part of a matrix
        gram[np.triu_indices(2 * self.p)] = parallel_map(
            task=self._Gij,
            values=list(
                combinations_with_replacement(range(2 * self.p), r=2)
            ),  # all combinations of indices, including the diagonal
            task_args=(delta,),
        )
        # use that the gram matrix is hermetian
        # np.tril_indices returns the indices of the lower triangular part of a matrix, -1 to exclude the diagonal
        gram = gram + np.conj(gram.T) - np.diag(gram.diagonal())
        return np.matrix(gram)

    def _grad_element(self, i, delta: tuple[float]) -> np.complex_:
        """Evaluate the gradient element <d_i Psi|H|Psi> of the qaoa state wrt the i-th parameter.
        TODO: maybe this can be done more efficiently by summing the products of propagators and multiplying the result with mixer_ground. Corresponds to pulling | - > out to the right.

        Args:
            i (_type_): _description_
            delta (tuple[float]): _description_

        Returns:
            np.complex_: _description_
        """
        if i <= self.p - 1:
            # compute the left state <d_i Psi| = (sum_i(U_i)|psi>)
            left_state = sum(  # the sum over all parts of B
                self.circuitDiff(
                    delta, [Gate("X", [k])], 2 * i + 2
                ).run(  # insert X after B, positions are handpicked and correspond to gate indices not qaoa layers
                    self.mixer_ground
                )
                for k in range(self.n)
            )
            return (left_state.dag() * self.H * self.state(delta))[0, 0]

        if i > self.p - 1:
            left_state = sum(
                (
                    self.qubo[k, k] + self.qj[k]
                )  # the linear qubo coefficients coming from Z_k
                * (
                    self.circuitDiff(delta, [Gate("Z", [k])], 2 * i + 1).run(
                        self.mixer_ground
                    )
                )
                for k in range(self.n)
            ) + sum(
                2  # factor of two because qubo is symmetric and we only add each gate combination once
                * self.qubo[k, l]  # the quadratic qubo coefficients coming from Z_k Z_l
                * (
                    self.circuitDiff(
                        delta, [Gate("Z", [k]), Gate("Z", [l])], 2 * i + 1
                    ).run(self.mixer_ground)
                )
                for k, l in combinations(range(self.n), 2)
            )
            return (left_state.dag() * self.H * self.state(delta))[0, 0]

    def grad(self, delta: tuple[float], **kwargs) -> NDArray:
        return np.matrix(
            parallel_map(self._grad_element, range(len(delta)), task_args=(delta,))
        )


class QAOAResult:
    def __init__(self) -> None:
        self._message = None
        self._optimizer_name = None
        self._sucess = None
        self._parameter_path = None
        self._optimal_fun_value = None
        self._num_fun_calls = None
        self._num_steps = None
        self._duration = None
        self._optimal_state = None
        self._optimal_parameters = None
        self._orig_result = None

    # orig_result is the result of the optimizer
    @property
    def orig_result(self):
        """The orig_result property."""
        return self._orig_result

    @orig_result.setter
    def orig_result(self, value):
        self._orig_result = value

    @property
    def qaoa(self) -> QAOA:
        """The qaoa
        property."""
        return self._qaoa

    @qaoa.setter
    def qaoa(self, value: QAOA):
        self._qaoa = value

    @property
    def parameters(self) -> tuple[float]:
        """The optimal_parameters property."""
        return self._optimal_parameters

    @parameters.setter
    def parameters(self, value: tuple[float]):
        self._optimal_parameters = value

    @property
    def state(self) -> Qobj():
        """The optimal_state property."""
        return self._optimal_state

    @state.setter
    def state(self, value: Qobj()):
        self._optimal_state = value

    @property
    def duration(self) -> float:
        """The duration property."""
        return self._duration

    @duration.setter
    def duration(self, value: float):
        self._duration = value

    @property
    def num_steps(self) -> int | None:
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
    def value(self) -> float:
        """The optimal_fun_value property."""
        return self._optimal_fun_value

    @value.setter
    def value(self, value: float):
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

    def prob(self) -> float:
        """The distance from the groundstate property."""
        eigenstates = self.qaoa.H.eigenstates()
        groundstates = eigenstates[1][np.where(eigenstates[0] == eigenstates[0][0])]
        return max(abs(self.state.overlap(ground)) ** 2 for ground in groundstates)

    def __str__(self):
        return f"""
        {self.optimizer_name} terminated with {'no' if not self.success else ''} sucess with message
        \"{self.message}\"

        optimal parameters:     {self.parameters}
        optimal value:          {self.value}
        number of fun calls:    {self.num_fun_calls}
        number of steps:        {self.num_steps}
        """


def scipy_optimize(
    qaoa: QAOA,
    delta_0: tuple[float],
    max_iter: int = 1000,
) -> QAOAResult:
    opt_result = QAOAResult()
    t_0 = time()
    min_result = minimize(
        qaoa.expectation,
        x0=np.array(delta_0),
        method="COBYLA",
        options={"maxiter": max_iter},
    )
    opt_result.orig_result = min_result
    dt = time() - t_0
    opt_result.qaoa = qaoa
    opt_result.duration = dt
    opt_result.success = min_result.success
    opt_result.parameters = min_result.x
    opt_result.message = min_result.message
    opt_result.value = min_result.fun
    opt_result.num_fun_calls = min_result.nfev
    try:
        opt_result.num_steps = min_result.nit
    except AttributeError:
        pass
    opt_result.state = qaoa.state(opt_result.parameters)
    opt_result.optimizer_name = "scipy_cobyla"

    return opt_result


def finitediff(
    func: Callable[[tuple[float]], Qobj], epsilon: float = 1e-10
) -> Callable[[tuple[float]], list[Qobj]]:
    def dfunc(params) -> list:
        params = tuple(params)
        difference = list()
        num_pars = int(len(params))
        for i in range(num_pars):
            p_params = params[:i] + (params[i] + epsilon,) + params[i + 1 :]
            difference.append((func(p_params) - func(params)) / epsilon)
        return difference

    return dfunc


def gen_grad(
    pars: tuple[float],
    qaoa: QAOA,
) -> np.matrix:
    """Generate gradient for given parameters and QAOA instance. Uses finite differences.

    Args:
        pars (tuple[float]): Parameters for which to generate the gradient.
        qaoa (QAOA): QAOA instance.

    Returns:
        np.matrix: Gradient.
    """
    dpsi = finitediff(qaoa.state)
    p = int(len(pars))
    out = np.matrix(
        [(dpsi(pars)[k].dag() * qaoa.H * qaoa.state(pars))[0, 0] for k in range(p)]
    )
    return out


def gen_gram(pars: tuple[float], qaoa: QAOA) -> np.matrix:
    """Generate Gram matrix for given parameters and QAOA instance. Uses finite differences.

    Args:
        pars (tuple[float]): Parameters for which to generate the Gram matrix.
        qaoa (QAOA): QAOA instance.

    Returns:
        np.matrix: Gram matrix.
    """
    dpsi = finitediff(qaoa.state)
    num_pars = int(len(pars))
    return np.matrix(
        [
            [((dpsi(pars)[j]).dag() * dpsi(pars)[k])[0, 0] for k in range(num_pars)]
            for j in range(num_pars)
        ]
    )  # order of j,k must be correct -> j should be rows, k should be columns


def gen_tdvp_rhs(
    t: float, x: Tuple[float], qaoa: QAOA
) -> NDArray:  # right hand side of linear equation system of tdvp. In the right format for the scipy solvers.
    """right hand side of linear equation system of tdvp. In the right format for the scipy solvers. Uses finite differences.

    Args:
        t (float): time variable
        x (tuple[float]): parameter input

    Returns:
        NDArray: the matrix defining the RHS of the equation system
    """
    inv_real_gram = linalg.inv(2 * np.real(gen_gram(x, qaoa)))
    real_grad = 2 * np.real(gen_grad(x, qaoa))
    return np.array(-inv_real_gram * real_grad.T).flatten()


def qaoa_tdvp_rhs(t: float, x: Tuple[float], qaoa: QAOA) -> NDArray:
    """right hand side of linear equation system of tdvp. In the right format for the scipy solvers.
    Uses the qaoa class to calculate the gradient and gram matrix.
    Inverts the gram matrix using the scipy linalg.inv.

    Args:
        t (float): time variable
        x (tuple[float]): parameter input

    Returns:
        NDArray: the matrix defining the RHS of the equation
    """
    inv_real_gram = linalg.inv(2 * np.real(qaoa.gram(x)))
    real_grad = 2 * np.real(qaoa.grad(x))
    return np.array(-inv_real_gram * real_grad.T).flatten()


def qaoa_lineq_tdvp_rhs(t: float, x: Tuple[float], qaoa: QAOA) -> np.ndarray:
    """right hand side of linear equation system of tdvp. In the right format for the scipy solvers.
    Uses the qaoa class to calculate the gradient and gram matrix.
    Solves the linear equation system G_ijx_i = d_j E using the scipy linalg.solve in order to avoid inverting the gram matrix.

    Args:
        t (float): time variable
        x (tuple[float]): parameter input

    Returns:
        NDArray: the matrix defining the RHS of the equation
    """
    gram: NDArray = 2 * np.real(qaoa.gram(x))
    grad: NDArray = 2 * np.real(qaoa.grad(x)).T
    return linalg.solve(
        gram,
        -grad,
        overwrite_a=True,
        overwrite_b=True,
        assume_a="her",  # "pos", "sym", "gen"
    ).flatten()


def tdvp_optimize_qaoa(
    qaoa: QAOA,
    delta_0: tuple[float],
    Delta: float = 0.01,
    rhs_mode: str = "qaoa",
    int_mode: str = "RK45",
    grad_tol: float = 1e-6,
    max_iter: int = 1000,
) -> QAOAResult:  # rhs_mode: "qaoa", "lineq", "lineq_qaoa"
    """optimize an qaoa instance by tdvp for imaginary time evolution.

    Args:
        qaoa (QAOA): the qaoa instance to be optimized
        delta_0 (tuple[float]): the initial parameters
        Delta (float, optional): The imaginary time duration for which to evolve. Defaults to .01.
        rhs_mode (str, optional): Mode for computing the rhs of the ODE. Possible values are "qaoa","gen" and "qaoa_lineq".
                                "qaoa" uses the qaoa instance to compute the rhs, "gen" uses the gen_grad and gen_gram.
                                "qaoa_lineq" uses the qaoa instance to compute the rhs, but does not invert the gram matrix.
                                Instead it solves the linear equation system G_ijx_i = d_j E, where x_i is the time derivative
                                of the parameters. Defaults to "qaoa".
        int_mode (str, optional): Mode for solving the ODE. Possible values are "RK45", "RK23", "DOP853", "euler".

    Returns:
        QAOAResult: the result of the qaoa after optimization.
    """
    rhs_step = 0
    if rhs_mode == "qaoa":

        def tdvp_rhs(t, x) -> NDArray:
            nonlocal rhs_step
            rhs_step += 1
            print(f"rhs step {rhs_step}", end="\r")
            return qaoa_tdvp_rhs(t, x, qaoa)

    elif rhs_mode == "gen":

        def tdvp_rhs(t, x) -> NDArray:
            nonlocal rhs_step
            rhs_step += 1
            print(f"rhs step {rhs_step}", end="\r")
            return gen_tdvp_rhs(t, x, qaoa)

    elif rhs_mode == "qaoa_lineq":

        def tdvp_rhs(t, x) -> NDArray:
            nonlocal rhs_step
            rhs_step += 1
            print(f"rhs step {rhs_step}", end="\r")
            return qaoa_lineq_tdvp_rhs(t, x, qaoa)

    def termination_event(t, x) -> float:
        return linalg.norm(qaoa.grad(x))

    termination_event.terminal = True

    match int_mode:
        case "euler":
            t_0 = process_time()
            delta = delta_0
            while linalg.norm(qaoa.grad(delta)) > grad_tol and rhs_step < max_iter:
                delta = delta + Delta * tdvp_rhs(
                    0, delta
                )  # tdvp_rhs increases rhs_step by 1
            dt = process_time() - t_0  # time for integration
            result = QAOAResult()
            result.orig_result = None
            result.success = (
                linalg.norm(qaoa.grad(delta)) < grad_tol
            )  # success of integration
            result.parameters = delta  # last step
            result.message = f"Euler solver terminated with {'success' if result.success else 'no success'} after {rhs_step} steps."  # message from the solver
            result.num_fun_calls = rhs_step  # number of function calls

        case _:
            t_0 = process_time()
            int_result = integrate.solve_ivp(
                fun=tdvp_rhs,
                t_span=(0, Delta),
                y0=delta_0,
                method=int_mode,
                events=termination_event,
                max_iter=max_iter,
            )
            dt = process_time() - t_0  # time for integration

            result = QAOAResult()  # create result object
            result.orig_result = int_result
            result.success = int_result.success  # success of integration
            result.parameters = tuple(par[-1] for par in int_result.y)  # last step
            result.message = int_result.message  # message from the solver
            result.num_fun_calls = int_result.nfev  # number of function calls

    result.qaoa = qaoa
    result.duration = dt  # time for integration
    result.num_steps = rhs_step  # number of steps
    result.optimizer_name = f"tdvp_optimizer with {'circuit' if rhs_mode else 'finitediff'} gradient evaluation and {int_mode} as integration mode."
    result.state = qaoa.state(result.parameters)  # final state
    result.value = qaoa.expectation(
        result.parameters
    )  # expectation value of the optimal state

    return result
