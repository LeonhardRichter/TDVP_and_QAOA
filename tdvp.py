#%%
import numpy as np
from numpy import pi
from numpy.typing import NDArray, ArrayLike, DTypeLike

from itertools import combinations
from typing import Callable, Iterable

import scipy as sp
from scipy import linalg
from scipy import integrate
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import plotly.express as px

from qutip import *
from qutip.qip.operations import *
from qutip.qip.circuit import QubitCircuit, Gate

from qaoa import qaoa_circuit

#%%

def sz(n:int,i:int):
    return expand_operator(sigmaz(),n,i,[2 for _ in range(n)])

def sx(n:int,i:int):
    return expand_operator(sigmax(),n,i,[2 for _ in range(n)])
    
minus = (basis(2,0)-basis(2,1)).unit()
def rzz(arg_value):
    return tensor(rz(arg_value),rz(-arg_value))

class tdvp_result():
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



class tdvp_optimizer():
    def __init__(self, psi:Callable[[tuple[float]],Qobj], H:Qobj, qubo:NDArray) -> None:
        self._psi = psi 
        self._H = H
        assert qubo.T.all()==qubo.all(), "Qubo matrix not symmetric" 
        self._qubo = qubo
        self.n = len(H.dims[0])

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

    def flow(self,pars, stepsize) -> tuple:
        """flow according to tdvp from pars for stepsize duration.

        Args:
            pars (tuple): starting point parameters
            stepsize (float): stepsize of the flow

        Returns:
            _type_: _description_
        """
        pass

    def optimize(self,tol, stepsize) -> tdvp_result:
        """Run the optimization algorithm using the tdvp flow.

        Args:
            tol (float): _description_
            stepsize (float): _description_

        Returns:
            tdvp_result: _description_
        """
        pass

    def get_qaoa_gram(self,pars:dict):
        """evaluate the gram matrix on a quantum circuit.

        Args:
            pars (dict): dict with 'gammas':tuple and 'betas':tuple

        Returns:
            _type_: _description_
        """
        p = int(len(pars)/2)
        def U_i(pars:dict,oper:Gate,i:int):
            if self.qubo is not None:
                # define what to apply to the circuit in each H_p turn
                def qcH(gamma:float) -> QubitCircuit:
                    qc = QubitCircuit(self.n)
                    qc.user_gates = {"RZZ",rzz}
                    for j in range(self.n):
                        qc.add_gate("RZ",targets=j, arg_value=2*gamma*self.qubo[j][j],arg_label=f"2*{round(gamma,2)}*{self.qubo[j][j]}")
                    for j,k in combinations(range(self.n),2):
                        qc.add_gate("RZZ", targets=[j,k], arg_value=2*gamma*self.qubo[j][k],arg_label=f"2*{round(gamma,2)}*{self.qubo[j][k]}")
                    return qc
            else:
                assert self.H.isherm
                def qcH(gamma:float) -> QubitCircuit:
                    def H_exp(arg_value):
                        return (-1j*arg_value*self.H).expm()
                    qc = QubitCircuit(self.n)
                    qc.user_gates = {"H_exp":H_exp}
                    qc.add_gate("H_exp", arg_value=gamma,arg_label=f"{round(gamma,2)}")
                    return qc

            def qcB(beta:float) -> QubitCircuit:
                qc = QubitCircuit(self.n)
                qc.add_1q_gate("RX",arg_value=2*beta)
                return qc


            qc = QubitCircuit(self.n)
            for _ in range(p):
                qc.add_circuit(qcH(pars['gammas'][i]))
                qc.add_circuit(qcB(pars['betas'][i]))
                if _==i: qc.add_gate(oper)
            
            return qc

        def U_i_tilde(pars:dict,oper:Gate,i:int):
            if self.qubo is not None:
                # define what to apply to the circuit in each H_p turn
                def qcH(gamma:float) -> QubitCircuit:
                    qc = QubitCircuit(self.n)
                    qc.user_gates = {"RZZ",rzz}
                    for j in range(self.n):
                        qc.add_gate("RZ",targets=j, arg_value=2*gamma*self.qubo[j][j],arg_label=f"2*{round(gamma,2)}*{self.qubo[j][j]}")
                    for j,k in combinations(range(self.n),2):
                        qc.add_gate("RZZ", targets=[j,k], arg_value=2*gamma*self.qubo[j][k],arg_label=f"2*{round(gamma,2)}*{self.qubo[j][k]}")
                    return qc
            else:
                assert self.H.isherm
                def qcH(gamma:float) -> QubitCircuit:
                    def H_exp(arg_value):
                        return (-1j*arg_value*self.H).expm()
                    qc = QubitCircuit(self.n)
                    qc.user_gates = {"H_exp":H_exp}
                    qc.add_gate("H_exp", arg_value=gamma,arg_label=f"{round(gamma,2)}")
                    return qc

            def qcB(beta:float) -> QubitCircuit:
                qc = QubitCircuit(self.n)
                qc.add_1q_gate("RX",arg_value=2*beta)
                return qc

            def oper_gate():
                return oper

            qc = QubitCircuit(self.n)
            for _ in range(p):
                qc.add_circuit(qcH(pars['gammas'][i]))
                if _==i: qc.add_gate(oper)
                qc.add_circuit(qcB(pars['betas'][i]))
            
            return qc

        # for the adjont circuits use U_i(-pars,oper=RX(-pi/2)).reverse() and so on

        return U_i

# %%
def psi(pars:tuple[float])->Qobj:
    return tensor([np.cos(pars[0]/2)*basis(2,0)+np.exp(1j*pars[1])*np.sin(pars[0]/2)*basis(2,1) for _ in range(2)])
            
tdvp_ = tdvp_optimizer(psi,tensor(sigmaz(),sigmaz()),np.array([[1,2],[2,1]]))
U_i = tdvp_.get_qaoa_gram({"betas":(1,1),"gammas":(1,1)})
U_i(pars={"betas":(1,1,1,1),"gammas":(1,1,1,1)},oper=Gate("X",[1]),i=0).reverse_circuit()
# %

# %%
