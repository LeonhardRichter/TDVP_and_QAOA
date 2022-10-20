#%%
from math import *
import numpy as np
import random as rng
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from typing import Union
from qaoa_and_tdvp import QAOA, tdvp_optimize_qaoa, scipy_optimize, QAOAResult, Qobj
from MaxCut import MaxCut
from itertools import *

from qutip import tensor, basis, Qobj
from qutip.qip.operations import expand_operator, rz

from benchmark import (
    get_all_connected,
    get_connected_rn_graph,
    get_rn_qubo,
    select_if_connected,
    Benchmark,
)
from qutip.parallel import parallel_map, serial_map

# %%
rng.sample
np.eye
np.all
np.isclose
