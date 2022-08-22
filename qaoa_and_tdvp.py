#%%
from lib2to3.pgen2.token import OP
from time import time as ttime
from abc import ABC, abstractmethod, abstractproperty
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
class QAOAResult():
    def __init__(self) -> None:
        pass
    
    @property
    def optimal_parameters(self)->tuple[float]:
        """The optimal_parameters property."""
        return self._optimal_parameters

    @optimal_parameters.setter
    def optimal_parameters(self, value:tuple[float]):
        self._optimal_parameters= value

    @property
    def optimal_state(self)->Qobj():
        """The optimal_state property."""
        return self._optimal_state
    @optimal_state.setter
    def optimal_state(self, value:Qobj()):
        self._optimal_state = value

    @property
    def duration(self)->float:
        """The duration property."""
        return self._duration
    @duration.setter
    def duration(self, value:float):
        self._duration = value

    @property
    def num_steps(self)->int:
        """The num_steps property."""
        return self._num_steps
    @num_steps.setter
    def num_steps(self, value:int):
        self._num_steps = value
    @property
    def num_fun_calls(self)->int:
        """The num_fun_calls property."""
        return self._num_fun_calls
    @num_fun_calls.setter
    def num_fun_calls(self, value:int):
        self._num_fun_calls = value

    @property
    def optimal_fun_value(self)->float:
        """The optimal_fun_value property."""
        return self._optimal_fun_value
    @optimal_fun_value.setter
    def optimal_fun_value(self, value:float):
        self._optimal_fun_value = value

    @property
    def parameter_path(self)->list[tuple[float]]:
        """The parameter_path property."""
        return self._parameter_path
    @parameter_path.setter
    def parameter_path(self, value:list[tuple[float]]):
        self._parameter_path = value

    @property
    def sucess(self)->bool:
        """The sucess property."""
        return self._sucess
    @sucess.setter
    def sucess(self, value:bool):
        self._sucess = value

    @property
    def optimizer_name(self)->str:
        """The optimizer_name property."""
        return self._optimizer_name
    @optimizer_name.setter
    def optimizer_name(self, value:str):
        self._optimizer_name = value


    def __str__(self):
        return f"""
        {self.optimizer_name} terminated with {'no' if not self.sucess else''} sucess.
        optimal parameters:     {self.optimal_parameters}
        optimal value:          {self.optimal_fun_value}
        number of fun calls:    {self.num_fun_calls}
        number of steps:        {self.num_steps}
        """


class Optimizer(ABC):
    def __init__(self) -> None:
        pass

    @abstractproperty
    def name(self)->str:
        return ''

    @abstractmethod
    def optimize(self,fun:Callable[[tuple[float]],float],detla_0:tuple[float])->QAOAResult:
        pass

#%%
class ScipyOptimizter(Optimizer):
    def __init__(self) -> None:
        super().__init__()
    @property
    def name(self)->str:
        return'ScipyOptimizer'
    
    def optimize(self,fun:Callable[[tuple[float]],float],delta_0:tuple[float])->QAOAResult:
        opt_result = QAOAResult()
        t_0 = ttime()
        min_result = minimize(fun, x0=delta_0, method="COBYLA")
        dt = ttime() - t_0 
        opt_result.duration = dt
        opt_result.sucess = min_result.sucess
        opt_result.optimal_parameters = min_result.x
        opt_result.optimal_fun_value = min_result.fun
        opt_result.num_fun_calls = min_result.nfev
        opt_result.num_steps = min_result.nit
        opt_result.optimizer_name = self.name

        return opt_result


#%%

class QAOA():
    def __init__(self,
                 hamiltonian: Qobj = None,
                 hamiltonian_ground: Qobj = None,
                 qubo:ArrayLike=None,
                 mixer: Qobj = None,
                 mixer_ground:Qobj=None,
                 p:int=1,
                 optimizer:Optimizer = None) -> None:

        self._H = hamiltonian
        self.H_ground = hamiltonian_ground
        self.qubo = qubo
        self.p = p

        if self.qubo is not None:
            self._n=qubo.shape[0]
        if self.H is not None:
            self._n = len(self.H.dims[0])

        self._optimzer = optimizer

        self.mixer = mixer
        if mixer_ground is not None:
            self.mixer_ground = mixer_ground
        else:
            self.mixer_ground = tensor([minus for _ in range(self.n)])
    # hamiltonian_ground
    @property
    def H_ground(self)->Qobj:
        """The H_ground property."""
        return self._H_ground
    @H_ground.setter
    def H_ground(self, value:Qobj):
        self._H_ground = value

    # mixer ground
    @property
    def mixer_ground(self)->Qobj:
        """The mxier_ground property."""
        return self._mixer_ground
    @mixer_ground.setter
    def mixer_ground(self, value:Qobj):
        self._mixer_ground = value
    # H
    @property
    def H(self)->Qobj:
        """The H property."""
        if self._H==None:
            if self.qubo is not None:
                self._H = H_from_qubo(self.qubo)
        return self._H
    @H.setter
    def H(self, value:Qobj):
        self._H = value
    # n
    @property
    def n(self)->int:
        """The n property."""
        if self._n==None:
            self._n==len(self.H.dims[0])
        return self._n
    @n.setter
    def n(self, value:int):
        self._n = value
    # optimizer
    @property
    def optimizer(self)->Optimizer:
        """The optimizer property."""
        return self._optimizer
    @optimizer.setter
    def optimizer(self, value:Optimizer):
        self._optimizer = value
    ##############################################################################################33
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

    
    # foo
    @property
    def foo(self):->The  property.
        """The foo property."""
        return self._foo
    @foo.setter
    def foo(self, value:The  property.):
        self._foo = value

    def expectation(self, delta: tuple[float]) -> float:
        assert len(delta)==2*self.p
        p = self.p
        qc = circuit(delta)
        n = self.n
        hamiltonian = self.H
        qaoa_state = qc.run(self.mixer_ground)
        return expect(hamiltonian, qaoa_state)

    def solve(self,delta_0:tuple[float])->QAOAResult:
        result = self.optimizer.optimize(self.expectation, delta_0)
        result.optimal_state = self.circuit(result.optimal_parameters).run(self.mixer_ground)
        return result