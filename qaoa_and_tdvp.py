#%%
from time import process_time, time
from abc import ABCMeta, abstractmethod, abstractproperty
from itertools import combinations, combinations_with_replacement
from typing import Callable, Iterable

import numpy as np
from numpy import pi
from numpy.typing import NDArray, ArrayLike, DTypeLike

import scipy as sp
from scipy import linalg
from scipy import integrate
from scipy.optimize import minimize, show_options

import matplotlib.pyplot as plt
import plotly.express as px

from qutip import *
from qutip.qip.operations import *
from qutip.qip.circuit import QubitCircuit, Gate

#%%
# Define general expressions and objects
def sz(n:int,i:int):
    return expand_operator(sigmaz(),n,i,[2 for _ in range(n)])

def sx(n:int,i:int):
    return expand_operator(sigmax(),n,i,[2 for _ in range(n)])

def rzz(arg_value):
    return tensor(rz(arg_value),rz(-arg_value))
    
minus = (basis(2,0)-basis(2,1)).unit()

def H_from_qubo(qubo: ArrayLike, constant: float = None) -> QobjEvo:
    n = qubo.shape[0]
    if constant == None:
        qconstant = Qobj(np.full((2**n, 2**n), 0),
                         dims=[[2 for _ in range(n)], [2 for _ in range(n)]])
    else:
        qconstant = constant*qeye([2 for _ in range(n)])
    H = sum([qubo[i][i]*sz(n, i) for i in range(n)])\
        + sum([qubo[j][k]*sz(n, j)*sz(n, k)for j, k in combinations(range(n), 2)]) \
        + qconstant
    return H

#%%
# Define qaoa class
class OptimizerResult(meta=ABCMeta):
    def __init__(self) -> None:
        pass

class Optimizer(meta=ABCMeta):
    def __init__(self) -> None:
        pass
    @abstractmethod
    def optimize(self,fun:Callable[[tuple[float]],float])->OptimizerResult:
        pass


class QAOA():
    def __init__(self,
                hamiltonian: Qobj = None,
                 qubo:ArrayLike=None,
                 mixer: Qobj = None,
                 p:int=1,
                 optimizer:Optimizer = None) -> None:
        self._H = hamiltonian
        self.qubo = qubo
        self.mixer = mixer
        self.p = p
        if self.qubo is not None:
            self._n=qubo.shape[0]
        if self.H is not None:
            self._n = len(self.H.dims[0])
        self._optimzer = optimizer
    # H
    @property
    def H(self):
        """The H property."""
        if self._H==None:
            if self.qubo is not None:
                self._H = H_from_qubo(self.qubo)
        return self._H
    @H.setter
    def H(self, value):
        self._H = value
    # n
    @property
    def n(self):
        """The n property."""
        if self._n==None:
            self._n==len(self.H.dims[0])
        return self._n
    @n.setter
    def n(self, value):
        self._n = value
    
    @property
    def optimizer(self):
        """The optimizer property."""
        return self._optimizer
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def circuit(self, delta:tuple[float]) -> QubitCircuit:
        assert len(delta)==2*self.p
        p = self.p
        n = self.n
        betas = delta[:p]
        gammas = delta[p:2*p]
        hamiltonian = self.H
        qubo = self.qubo
        mixer = self.mixer
        # # check input mode, prefere qubo mode
        # if linears is not None and quadratics is not None:
        #     linears = np.array(linears)
        #     quadratics = np.array(quadratics)
        #     qubo = quadratics.copy()
        #     linears += quadratics.diagonal()
        #     np.fill_diagonal(qubo,linears)
        # enter if either qubo input or linears and quadratics were given
        if qubo is not None:
            qubo = np.array(qubo)
            # define what to apply to the circuit in each H_p turn
            def qcH(gamma:float) -> QubitCircuit:
                qc = QubitCircuit(n)
                qc.user_gates = {"RZZ",rzz}
                for j in range(n):
                    qc.add_gate("RZ",targets=j, arg_value=2*gamma*qubo[j][j],arg_label=f"2*{round(gamma,2)}*{qubo[j][j]}")
                for j,k in combinations(range(n),2):
                    qc.add_gate("RZZ", targets=[j,k], arg_value=2*gamma*qubo[j][k],arg_label=f"2*{round(gamma,2)}*{qubo[j][k]}")
                return qc
        # if no qubo data is given but a Qobj hamiltonian, exponiate it
        elif hamiltonian is not None:
            assert hamiltonian.isherm, "hamiltonian must be hermetian"
            def qcH(gamma:float) -> QubitCircuit:
                def H_exp(arg_value):
                    return (-1j*arg_value*hamiltonian).expm()
                qc = QubitCircuit(n)
                qc.user_gates = {"H_exp":H_exp}
                qc.add_gate("H_exp", arg_value=gamma,arg_label=f'{round(gamma,2)}')
                return qc

        assert n>0
        assert qcH is not None

        # define mixer circuit
        if mixer is None:
            def qcB(beta:float) -> QubitCircuit:
                qc = QubitCircuit(n)
                qc.add_1q_gate("RX",arg_value=2*beta,arg_label=f'2*{round(beta,2)}')
                return qc
        else:
            initial_state_list = None
            print("Computing groundstate of mixer")
            initial_state = mixer.groundstate()[1]
            def B_exp(arg_value):
                    return (-1j*arg_value*mixer).expm()
            def qcB(beta:float) -> QubitCircuit:
                qc = QubitCircuit(n)
                qc.user_gates = {"B_exp":B_exp}
                qc.add_gate("B_exp", arg_value=beta, arg_label=f'2*{round(beta,2)}')


        qc = QubitCircuit(n)

        for i in range(p):
            qc.add_circuit(qcH(gammas[i]))
            qc.add_circuit(qcB(betas[i]))

        return qc

    
    def expectation(self, delta: tuple[float], mixer_ground=None) -> float:
        assert len(delta)==2*self.p
        p = self.p
        qc = circuit(delta)
        n = self.n
        hamiltonian = self.H
        if mixer_ground is None:
            mixer_ground = tensor([minus for _ in range(n)])
        result = qc.run(mixer_ground)
        return expect(hamiltonian, result)

    def qaoa_solver(qubo,p)->OptimizerResult:
        
        # opt_result = minimize(
        #     lambda pars:f(pars,H,ground,qubo=qubo),x0=tuple((0 for _ in range(2*p))),method="COBYLA"
        #     )

        return opt_result
