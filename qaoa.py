# %%
from abc import ABCMeta, abstractmethod, abstractproperty
from ast import arg
from asyncio import BaseTransport
from typing import Union, Iterable, Callable\

from itertools import combinations

from qutip import *
from qutip.qip.operations import *
from qutip.qip.circuit import QubitCircuit, Gate

import numpy as np
from numpy.typing import ArrayLike
from numpy import pi

import scipy as sp
import scipy.integrate as spint
import scipy.linalg as splinalg
from scipy.optimize import minimize, show_options

import plotly.express as px

import matplotlib.pyplot as plt

# %% [markdown]
# Implemet simple ´qaoa´ solver for general Hamiltonian given as $Z$-polynomial
# 
# $$
# 
#     H = C + \sum_{i}c_{i}Z_{i} + \sum_{i,j}c_{i,j}Z_{i}Z_{j} + \sum_{i,j,k}c_{i,j,k}Z_{i}Z_{j}Z_{k} + \dots
# 
# $$
# \\
# For that write global functions for all parts of `qaoa` $\rightarrow$ *trial state preperation*, *measurement*, computing the *expectation* .
# Write custom, abstract `OPTIMIZER` class wich can be substituted by scipy optimizers like `cobyla` and then later by the `tdvp optimizer`. 

# %%
def sz(n:int,i:int):
    return expand_operator(sigmaz(),n,i,[2 for _ in range(n)])

def sx(n:int,i:int):
    return expand_operator(sigmax(),n,i,[2 for _ in range(n)])
    
H = tensor(sigmaz(),sigmaz(),sigmaz())
B = sum([sx(3,_) for _ in range(3)])
minus = (basis(2,0)-basis(2,1)).unit()
minus_minus = tensor(minus,minus)

def rzz(arg_value):
    return tensor(rz(arg_value),rz(-arg_value))



# %%

def qaoa_circuit(betas: list[float], gammas: list[float],
                 hamiltonian: Qobj = None,
                 qubo:ArrayLike=None, linears:ArrayLike=None, quadratics:ArrayLike=None, constant:float=None,
                 mixer: Qobj = None) -> QubitCircuit:
    n = 0
    assert len(betas)==len(gammas)
    p = len(betas)

    # check input mode, prefere qubo mode
    if linears is not None and quadratics is not None:
        linears = np.array(linears)
        quadratics = np.array(quadratics)
        qubo = quadratics.copy()
        linears += quadratics.diagonal()
        np.fill_diagonal(qubo,linears)
    # enter if either qubo input or linears and quadratics were given
    if qubo is not None:
        qubo = np.array(qubo)
        n = qubo.shape[0]
        # define what to apply to the circuit in each H_p turn
        def qcH(gamma:float) -> QubitCircuit:
            qc = QubitCircuit(n)
            qc.user_gates = {"RZZ",rzz}
            for j in range(n):
                qc.add_gate("RZ",targets=j, arg_value=2*gamma*qubo[j][j],arg_label=f"2*{round(gamma,2)}*{qubo[j][j]}")
            for j,k in combinations(range(n),2):
                qc.add_gate("RZZ", targets=[j,k], arg_value=2*gamma*qubo[j][k],arg_label=f"2*{round(gamma,2)}*{qubo[j][k]}")
            if constant is not None:
                qc.add_gate("GLOBALPHASE",targets=range(n),arg_value=constant,arg_label=f"{round(constant,2)}")
            return qc
    # if no qubo data is given but a Qobj hamiltonian, exponiate it
    elif hamiltonian is not None:
        assert hamiltonian.isherm, "hamiltonian must be hermetian"
        n = len(hamiltonian.dims[0])
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


# %%
qubo = np.random.default_rng(seed=50505050).random(size=(3,3))
qubo = (qubo + qubo.T)*10
qubo

# %%
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

# %%
def f(pars: tuple[float], hamiltonian, mixer_ground=None, **kwargs) -> float:
    p = int(len(pars)/2)
    qc = qaoa_circuit(betas=pars[0:p], gammas=pars[p:2*p], **kwargs)
    n = qc.N
    if mixer_ground is None:
        mixer_ground = tensor([minus for _ in range(n)])
    result = qc.run(mixer_ground)
    return expect(hamiltonian, result)

# %%
f((1,1,1,1,1,1),hamiltonian=H, qubo=qubo, mixer_ground=tensor(minus,minus,minus))

# %%
class OPTIMIZER(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass
    @abstractproperty
    def name()-> str:
        return "abstract optimizer"

    @abstractmethod
    def optimize(fun, pars_0, *args):
        pass


# %%
opt_result = minimize(lambda pars:f(pars,H,tensor(minus,minus,minus),qubo=qubo),x0=tuple((0 for _ in range(2*10))),method="COBYLA")

# %%
opt_result

# %%
(qaoa_circuit(opt_result.x[:int(len(opt_result.x)/2)],opt_result.x[int(len(opt_result.x)/2):],qubo=qubo).run(state=tensor(minus,minus,minus)) - H.groundstate()[1]).norm()

# %%
def qaoa_solver(qubo,p):
    qubo = np.array(qubo)
    n = qubo.shape[0]
    H = H_from_qubo(qubo)
    ground = tensor([minus for _ in range(n)])

    opt_result = minimize(lambda pars:f(pars,H,ground,qubo=qubo),x0=tuple((0 for _ in range(2*p))),method="COBYLA")

    return opt_result
    

# %%
qaoa_solver(qubo,10)

# %%



