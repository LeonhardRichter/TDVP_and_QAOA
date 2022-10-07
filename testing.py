#%%
from qaoa_and_tdvp import *
from MaxCut import *

p = 2
instance = MaxCut(nx.triangular_lattice_graph(1, 1))
# nx.draw(instance.graph, with_labels=True)

qaoa = QAOA(qubo=instance.qubo, p=p, mapping=serial_map)
delta = tuple(1 for _ in range(2 * p))

#%%
qaoa.circuit(delta)

#%%
res = tdvp_optimize_qaoa(
    qaoa,
    delta,
    0.1,
    int_mode="euler",
    rhs_mode="qaoa",
    grad_tol=0.05,
    max_iter=50,
)
#
# sci_res = scipy_optimize(qaoa, delta)

# %%
