# vscode-fold=2
from time import time as time, time
from itertools import combinations, product, combinations_with_replacement
from typing import Callable, Tuple, Iterable

# import scipy as sp
from scipy import linalg
from scipy import integrate
from scipy.optimize import minimize

import numpy as np
from numpy.typing import NDArray

from qutip.parallel import parallel_map, serial_map
from qutip import expect, Qobj, tensor

# from qutip.qip.operations import expand_operator, rz
from qutip.qip.circuit import QubitCircuit, Gate
from qutip.ipynbtools import parallel_map as ipy_parallel_map

from quantum import H_exp, minus, q_j, rzz, sx, H_from_qubo


class QAOA:
    def __init__(
        self,
        qubo: NDArray,
        hamiltonian: Qobj = None,
        hamiltonian_ground: Qobj = None,
        p: int = 1,
        grammode: str = "single",
        mapping=parallel_map,
    ) -> None:

        self._H = hamiltonian
        self.H_ground = hamiltonian_ground
        self._qubo = qubo
        self.p = p

        self._n = qubo.shape[0]
        self._qj = q_j(qubo)
        assert self._n == len(self.H.dims[0])

        self.mixer_ground = tensor([minus for _ in range(self.n)])

        # # set the H qubo-circuit depending on inputmode
        # if use_hamiltonian:
        #     self._qcH = self._qcHhamiltonian
        #     self._qcB = self._qcBexp
        # else:
        #     self._qcH = self._qcHqubo
        #     self._qcB = self._qcBgates

        match grammode:
            case "double":
                self._A = self._Adouble
            case "single":
                self._A = self._Asingle

        assert mapping in {parallel_map, serial_map, ipy_parallel_map}
        self._mapping = mapping

        self.mixer = sum(sx(self.n, j) for j in range(self.n))

        self._num_gates = 0

    # qubo circuit
    @property
    def qubo(self) -> NDArray:
        """The qubo matrix of the problem."""
        return self._qubo

    @qubo.setter
    def qubo(self, value: NDArray) -> None:
        self._qubo = value
        self.H = H_from_qubo(value)
        self.n = value.shape[0]
        self.qj = q_j(value)
        self.mixer_ground = tensor([minus for _ in range(self.n)])
        self.mixer = sum(sx(self.n, j) for j in range(self.n))

    # num_gates
    @property
    def num_gates(self) -> int:
        """The number of gates applied in one run of the circuit."""
        return self._num_gates

    @num_gates.setter
    def num_gates(self, value: int) -> None:
        self._num_gates = value

    # mapping = ipy_parallel_map
    @property
    def mapping(self) -> Callable:
        """The mapping property."""
        return self._mapping

    @mapping.setter
    def mapping(self, value: Callable):
        self._mapping = value

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
        if self._H_ground is None:
            self._H_ground = self.H.groundstate()[1]
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

    def reset_gate_counter(self) -> None:
        self.num_gates = 0

    # the qaoa parts as QubitCircuits
    def _qcH(self, gamma: float) -> QubitCircuit:
        qc = QubitCircuit(self.n)
        qc.user_gates = {
            "RZZ": rzz,
            "H_{exp}": lambda x: H_exp(x, self.H),
            "B_{exp}": lambda x: H_exp(x, self.mixer),
        }
        for j in range(self.n):
            self.num_gates += 1
            qc.add_gate(
                "RZ",
                targets=[j],
                arg_value=-gamma * (self.qubo[j][j] + self.qj[j]),
                arg_label=f"2*{round(gamma, 2)}*(Q_{{{j}{j}}}+Q_{j})",
            )
        for j, k in combinations(range(self.n), 2):
            self.num_gates += 1
            qc.add_gate(
                "RZZ",
                targets=[j, k],
                arg_value=1 / 2 * gamma * self.qubo[j][k],
                arg_label=f"2*{round(gamma, 2)}*Q_{{{j}{k}}}",
            )
        return qc

    def _qcB(self, beta: float) -> QubitCircuit:
        qc = QubitCircuit(self.n)
        qc.user_gates = {
            "RZZ": rzz,
            "H_{exp}": lambda x: H_exp(x, self.H),
            "B_{exp}": lambda x: H_exp(x, self.mixer),
        }
        self.num_gates += qc.N
        qc.add_1q_gate("RX", arg_value=2 * beta, arg_label=f"2*{round(beta, 2)}")
        return qc

    # the whole qaoa circuit
    def circuit(
        self,
        delta: tuple[float],
        insert_gates: None | Iterable[Gate] = None,
        at_layer: None | int = None,
        inbetween: None | bool = None,
        pop_layers: None | Tuple[int, int] = None,
    ) -> QubitCircuit:
        p, n = self.p, self.n

        # assertions
        assert len(delta) == 2 * p
        if at_layer is None:
            at_layer = p
        if pop_layers is not None:
            assert (
                pop_layers[0] > at_layer
            ), "can't pop layers before the inserted layer"

        # init circuit
        qc = QubitCircuit(n)
        qc.user_gates = {
            "RZZ": rzz,
            "H_{exp}": lambda x: H_exp(x, self.H),
            "B_{exp}": lambda x: H_exp(x, self.mixer),
        }
        # add the layers before the layer to be inserted
        # note that if no at_layer is given, this will run until layer == p
        for i in range(at_layer):
            qc.add_circuit(self._qcH(delta[i + p]))
            qc.add_circuit(self._qcB(delta[i]))
        # layer is now at_layer
        # when not at the end of the circuit, continue with adding the gates
        # insert gates to be inserted inbetween qaoa blocks
        # check whether to insert gates inbetween qaoa blocks or after the layer
        if at_layer < p:
            match inbetween:
                case True:
                    qc.add_circuit(self._qcH(delta[at_layer + p]))
                    for gate in insert_gates:
                        qc.add_gate(gate)
                    qc.add_circuit(self._qcB(delta[at_layer]))
                case False:
                    qc.add_circuit(self._qcH(delta[at_layer + p]))
                    qc.add_circuit(self._qcB(delta[at_layer]))
                    for gate in insert_gates:
                        qc.add_gate(gate)
            self.num_gates += len(insert_gates)
            # add the rest of the qaoa layers
            # check if we need to pop layers
            if pop_layers is not None:
                for i in range(at_layer + 1, p):
                    if i in range(*pop_layers):
                        continue
                    qc.add_circuit(self._qcH(delta[i + p]))
                    qc.add_circuit(self._qcB(delta[i]))
            else:
                for i in range(at_layer + 1, p):
                    qc.add_circuit(self._qcH(delta[i + p]))
                    qc.add_circuit(self._qcB(delta[i]))
        return qc

    # wrappers for results
    def state(self, delta: tuple[float]) -> Qobj:
        return self.circuit(delta).run(self.mixer_ground)

    def expectation(self, delta: tuple[float]) -> float:
        assert len(delta) == 2 * self.p
        return expect(self.H, self.state(delta))

    # methods for tdvp metric and gradient
    def _Adouble(
        self,
        left: tuple[tuple[Gate], int, bool],
        right: tuple[tuple[Gate], int, bool],
        delta,
        pop_layers: Tuple[int, int] = None,
    ) -> np.complex_:
        """compute one summand of G_ij, i.e. compute the overlap of two states left and right.
        Each of those is a QAOA state with some operators inserted at a certain position

        Args:
            left  (tuple): (left_gates:tuple[Gate],left_layer:int, inbetween) left_gates is the list of operators to insert, left_pos is the layer index
                            to insert at, inbetween determines whether the gates are inserted inbetween H and B or after B.
            right (tuple): (right_gates:tuple[Gate],right_layer:int, inbetween) see left
        """
        left_state = self.circuit(delta, *left, pop_layers=pop_layers).run(
            self.mixer_ground
        )
        right_state = self.circuit(delta, *right, pop_layers=pop_layers).run(
            self.mixer_ground
        )
        return left_state.overlap(right_state)

    def _Asingle(
        self,
        left: tuple[tuple[Gate], int, bool],
        right: tuple[tuple[Gate], int, bool],
        delta,
        pop_layers: Tuple[int, int] = None,
    ) -> np.complex_:
        """compute one summand of G_ij, i.e. compute the combined circuit of left and right side and run it on input state.

        Args:
            left  (tuple): (left_gates:tuple[Gate],left_pos:int,inbetween) positions are layer-wise
            right (tuple): (right_gates:tuple[Gate],right_pos:int,inbetween)
        """
        m_delta = tuple(-t for t in delta)  # negative parameters
        qc = self.circuit(delta, *right, pop_layers=pop_layers)
        qc.add_circuit(
            self.circuit(m_delta, *left, pop_layers=pop_layers).reverse_circuit()
        )  # negative of the delta parameters gives in this case the adjoint gates
        return self.mixer_ground.overlap(qc.run(self.mixer_ground))

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
                    delta=delta,
                    left=([Gate("X", [l])], i % p, False),  # insert X at layer i
                    right=([Gate("X", [k])], j % p, False),  # same as above
                    # remove gates that will cancel each other out due to the adjoint circuit. This only works when i<=j! That is handled by self.gram only computing the upper triangle with this method
                    pop_layers=(j + 1, p),
                )
                for k, l in product(
                    range(n), repeat=2
                )  # sum over all possible combinations of X-gates
            )

        elif i <= p - 1 < j:  # upper right corner of the matrix
            element = (-1 / 2) * sum(
                (qubo[l][l] + qj[l])  # the linear qubo coefficients coming from Z
                * A(
                    left=([Gate("X", [k])], i % p, False),  # insert X after B
                    right=([Gate("Z", [l])], j % p, True),  # insert Z inbetween H and B
                    delta=delta,
                    pop_layers=(j + 1, p),
                )
                for k, l in product(range(n), repeat=2)
            ) + (1 / 4) * sum(
                2  # 2* is due to qubo being symmetric and not upper triangular
                * qubo[l][m]  # the quadratic qubo coefficients coming from ZZ
                * A(
                    left=([Gate("X", [k])], i % p, False),  # insert X after B
                    right=(
                        [Gate("Z", [l]), Gate("Z", [m])],
                        j % p,
                        True,
                    ),  # insert two Z's inbetween H and B
                    delta=delta,
                    pop_layers=(
                        j + 1,
                        p,
                    ),  # remove gates that will cancel each other out due to the adjoint circuit.
                )
                for k, (l, m) in product(
                    range(n), combinations(range(n), r=2)
                )  # sum over all possible combinations (l<m) of Z-gates, different orderings are counted twice
            )

        # the case j<=p-1<i is handled by hermiticity, as it lies in the lower triangle of the matrix

        elif i > p - 1 and j > p - 1:  # lower right corner of the matrix
            element = (
                (1 / 4)
                * sum(
                    (qubo[k, k] + qj[k])  # the linear qubo coefficients coming from Z_k
                    * (
                        qubo[l, l] + qj[l]
                    )  # the linear qubo coefficients coming from Z_l
                    * A(
                        left=([Gate("Z", [k])], i % p, True),  # insert Z after H
                        right=([Gate("Z", [l])], j % p, True),  # insert Z after H
                        delta=delta,
                        pop_layers=(j + 1, p),
                    )
                    for k, l in product(
                        range(n), repeat=2
                    )  # sum over all possible combinations of Z-gates
                )
                - (1 / 8)
                * sum(
                    2  # 2* is due to qubo being symmetric and not upper triangular
                    * qubo[k, l]  # the quadratic qubo coefficients coming from Z_k Z_l
                    * (
                        qubo[m, m] + qj[l]
                    )  # the linear qubo coefficients coming from Z_m
                    * A(
                        left=(
                            [Gate("Z", [k]), Gate("Z", l)],
                            i % p,
                            True,
                        ),  # insert two Z's after H
                        right=([Gate("Z", [m])], j % p, True),  # insert Z after H
                        delta=delta,
                        pop_layers=(j + 1, p),
                    )
                    for (k, l), m in product(
                        combinations(range(n), r=2), range(n)
                    )  # sum over all possible combinations (k<l) of Z-gates and (m) X Gates, different orderings are counted twice
                )
                - (1 / 8)
                * sum(
                    2  # 2* is due to qubo being symmetric and not upper triangular
                    * (
                        qubo[k, k] + qj[k]
                    )  # the linear qubo coefficients coming from Z_k
                    * qubo[l, m]  # the quadratic qubo coefficients coming from Z_l Z_m
                    * A(
                        left=([Gate("Z", [k])], i % p, True),  # insert Z after H
                        right=(
                            [Gate("Z", [l]), Gate("Z", [m])],
                            j % p,
                            True,
                        ),  # insert two Z's after H
                        delta=delta,
                        pop_layers=(j + 1, p),
                    )
                    for k, (l, m) in product(
                        range(n), combinations(range(n), r=2)
                    )  # sum over all possible combinations (l<m) of Z-gates and (k) X Gates, different orderings are counted twice
                )
                + (1 / 16)
                * sum(
                    2  # 2* is due to qubo being symmetric and not upper triangular
                    * qubo[k, l]  # the quadratic qubo coefficients coming from Z_k Z_l
                    * 2  # 2* is due to qubo being symmetric and not upper triangular
                    * qubo[m, n]  # the quadratic qubo coefficients coming from Z_m Z_n
                    * A(
                        left=(
                            [Gate("Z", [k]), Gate("Z", [l])],
                            i % p,
                            True,
                        ),  # insert two Z's after H
                        right=(
                            [Gate("Z", [m]), Gate("Z", [n])],
                            j % p,
                            True,
                        ),  # insert two Z's after H
                        delta=delta,
                        pop_layers=(j + 1, p),
                    )
                    for (k, l), (m, n) in product(
                        combinations(range(3), r=2), repeat=2
                    )  # sum over all possible combinations (k<l) of Z-gates and (m<n) of Z-gates, different orderings are counted twice
                )
            )
        return element

    def gram(
        self,
        delta: tuple[float],
        # **kwargs #not sure if necessary
    ) -> NDArray:
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
        gram[np.triu_indices(2 * self.p)] = self.mapping(
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

        p, n = self.p, self.n

        if i <= p - 1:
            # compute the left state <d_i Psi| = (sum_i(U_i)|psi>)
            left_state = sum(  # the sum over all parts of B
                self.circuit(delta, [Gate("X", [k])], i % p, False).run(
                    self.mixer_ground
                )
                for k in range(n)
            )

        if i > p - 1:
            left_state = sum(
                (-1 / 2)
                * (
                    self.qubo[k, k] + self.qj[k]
                )  # the linear qubo coefficients coming from Z_k
                * (
                    self.circuit(delta, [Gate("Z", [k])], i % p, True).run(
                        self.mixer_ground
                    )
                )
                for k in range(n)
            ) + sum(
                2  # factor of two because qubo is symmetric and we only add each gate combination once
                * (1 / 4)
                * self.qubo[k, l]  # the quadratic qubo coefficients coming from Z_k Z_l
                * (
                    self.circuit(
                        delta, [Gate("Z", [k]), Gate("Z", [l])], i % p, True
                    ).run(self.mixer_ground)
                )
                for k, l in combinations(range(n), 2)
            )
        return (-1j * left_state).overlap(self.H * self.state(delta))
        # the -1j comes from the analytical expression.
        # In the gram matrix these phase disappears, but here in the gradient it does not.

    def grad(
        self,
        delta: tuple[float],
        # **kwargs #not sure if necessary
    ) -> NDArray:
        return np.matrix(
            self.mapping(self._grad_element, range(len(delta)), task_args=(delta,))
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
        self._prob = None
        self._num_gates = None

    # num_gates
    @property
    def num_gates(self) -> type:
        """The num_gates property."""
        return self._num_gates

    @num_gates.setter
    def num_gates(self, value: type) -> None:
        self._num_gates = value

    # orig_result is the result of the optimizer
    @property
    def orig_result(self):
        """The orig_result property."""
        return self._orig_result

    @orig_result.setter
    def orig_result(self, value) -> None:
        self._orig_result = value

    @property
    def qaoa(self) -> QAOA:
        """The qaoa
        property."""
        return self._qaoa

    @qaoa.setter
    def qaoa(self, value: QAOA) -> None:
        self._qaoa = value

    @property
    def parameters(self) -> tuple[float]:
        """The optimal_parameters property."""
        return self._optimal_parameters

    @parameters.setter
    def parameters(self, value: tuple[float]) -> None:
        self._optimal_parameters = value

    @property
    def state(self) -> Qobj():
        """The optimal_state property."""
        return self._optimal_state

    @state.setter
    def state(self, value: Qobj()) -> None:
        self._optimal_state = value

    @property
    def duration(self) -> float:
        """The duration property."""
        return self._duration

    @duration.setter
    def duration(self, value: float) -> None:
        self._duration = value

    @property
    def num_steps(self) -> int | None:
        """The num_steps property."""
        try:
            return self._num_steps
        except AttributeError:
            return None

    @num_steps.setter
    def num_steps(self, value: int) -> None:
        self._num_steps = value

    @property
    def num_fun_calls(self) -> int:
        """The num_fun_calls property."""
        return self._num_fun_calls

    @num_fun_calls.setter
    def num_fun_calls(self, value: int) -> None:
        self._num_fun_calls = value

    @property
    def value(self) -> float:
        """The optimal_fun_value property."""
        return self._optimal_fun_value

    @value.setter
    def value(self, value: float) -> None:
        self._optimal_fun_value = value

    @property
    def parameter_path(self) -> list[tuple[float]]:
        """The parameter_path property."""
        return self._parameter_path

    @parameter_path.setter
    def parameter_path(self, value: list[tuple[float]]) -> None:
        self._parameter_path = value

    @property
    def success(self) -> bool:
        """The sucess property."""
        return self._sucess

    @success.setter
    def success(self, value: bool) -> None:
        self._sucess = value

    @property
    def optimizer_name(self) -> str:
        """The optimizer_name property."""
        return self._optimizer_name

    @optimizer_name.setter
    def optimizer_name(self, value: str) -> None:
        self._optimizer_name = value

    # message
    @property
    def message(self) -> str:
        """The message property."""
        return self._message

    @message.setter
    def message(self, value: str) -> None:
        self._message = value

    # problem
    @property
    def prob(self) -> float:
        """The maximal overlap with one of the ground states."""
        if self._prob is None:
            eigenstates = self.qaoa.H.eigenstates()
            groundstates = eigenstates[1][np.where(eigenstates[0] == eigenstates[0][0])]
            self.prob = max(
                abs(self.state.overlap(ground)) ** 2 for ground in groundstates
            )
        return self._prob

    @prob.setter
    def prob(self, value: float) -> None:
        self._prob = value

    def __repr__(self) -> str:
        return f"""
        {self.optimizer_name} terminated with {'no ' if not self.success else ''}sucess with message
        \"{self.message}\"
        This took {self.duration:.2f} seconds
        
            optimal parameters: {self.parameters}
                 optimal value: {self.value}
        maximal ground overlap: {self.prob}
           number of fun calls: {self.num_fun_calls}
               number of steps: {self.num_steps}
               number of gates: {self.num_gates}
        """


def scipy_optimize(
    qaoa: QAOA,
    delta_0: tuple[float],
    max_iter: int = 1000,
) -> QAOAResult:
    qaoa.reset_gate_counter()
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
    opt_result.num_gates = qaoa.num_gates

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
    p = qaoa.p
    out = np.matrix(
        [(dpsi(pars)[k].overlap(qaoa.H * qaoa.state(pars))) for k in range(2 * p)]
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
            [((dpsi(pars)[j]).overlap(dpsi(pars)[k])) for k in range(num_pars)]
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
    _ = t
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
    _ = t
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
    _ = t
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
    grad_tol: float = 1e-3,
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
    assert rhs_mode in {
        "qaoa",
        "gen",
        "qaoa_lineq",
    }, "rhs_mode must be one of 'qaoa', 'gen' or 'qaoa_lineq'"

    qaoa.reset_gate_counter()

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

    def tdvp_terminal(t, x) -> float:
        return (
            linalg.norm(tdvp_rhs(t, x)) - grad_tol
        )  # stop if the norm of the rhs is smaller than grad_tol

    tdvp_terminal.terminal = True  # this is needed for the scipy solver

    match int_mode:
        case "euler":
            t_0 = time()
            delta = delta_0
            # perform the solving loop
            while rhs_step < max_iter:
                rhs = tdvp_rhs(0, delta)  # tdvp_rhs increases rhs_step by 1
                delta = delta + Delta * rhs
                if linalg.norm(rhs) < grad_tol:  # break when gradient is small enough
                    break
            dt = time() - t_0  # time for integration
            print("done\n")
            # save the result
            result = QAOAResult()
            result.orig_result = None
            result.success = (
                linalg.norm(qaoa.grad(delta)) < grad_tol
            )  # success of integration
            result.parameters = delta  # last step
            result.message = f"Euler solver terminated with \
            {'success' if result.success else 'no success'} after {rhs_step} steps."  # message from the solver
            result.num_steps = rhs_step  # number of steps

        case _:
            t_0 = time()
            # solve the ODE
            int_result = integrate.solve_ivp(
                fun=tdvp_rhs,
                t_span=(0, Delta),
                y0=delta_0,
                method=int_mode,
                events=tdvp_terminal,
            )
            print("\n done")
            dt = time() - t_0  # time for integration
            # save the result
            result = QAOAResult()  # create result object
            result.orig_result = int_result
            result.success = int_result.success  # success of integration
            result.parameters = tuple(par[-1] for par in int_result.y)  # last step
            result.message = int_result.message  # message from the solver
            result.num_fun_calls = int_result.nfev  # number of function calls
    # save the rest of the result, same for both sovlers
    result.qaoa = qaoa
    result.duration = dt  # time for integration
    result.num_steps = rhs_step  # number of steps
    result.optimizer_name = f"tdvp_optimizer with {'circuit' if rhs_mode else 'finitediff'} gradient evaluation and {int_mode} as integration mode"
    result.state = qaoa.state(result.parameters)  # final state
    result.value = qaoa.expectation(
        result.parameters
    )  # expectation value of the optimal state
    result.num_gates = qaoa.num_gates  # number of gates

    qaoa.reset_gate_counter()

    return result
