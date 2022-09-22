#%%
from qaoa_and_tdvp import *
from MaxCut import *

p = 3
instance = MaxCut(nx.house())
# nx.draw(instance.graph, with_labels=True)

qaoa = QAOA(qubo=instance.qubo, p=p)
delta = tuple(1 for _ in range(2 * p))

#%%
res = tdvp_optimize_qaoa(
    qaoa,
    delta,
    0.1,
    int_mode="euler",
    rhs_mode="qaoa_lineq",
    grad_tol=0.05,
    max_iter=300,
)
# %%
sci_res = scipy_optimize(qaoa, delta)
