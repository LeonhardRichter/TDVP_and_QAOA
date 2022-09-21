#%%
from qaoa_and_tdvp import *
from MaxCut import *

p = 3
instance = MaxCut(nx.triangular_lattice_graph(1, 1))
# nx.draw(instance.graph, with_labels=True)

qaoa = QAOA(qubo=instance.qubo, p=p)
delta = tuple(1 for _ in range(2 * p))

#%%
gram = qaoa.gram(delta)
grad = qaoa.grad(delta)
g_gram = gen_gram(delta, qaoa)
g_grad = gen_grad(delta, qaoa)

#%%
res = tdvp_optimize_qaoa(
    qaoa, delta, 0.1, int_mode="euler", rhs_mode="gen", grad_tol=0.05, max_iter=300
)

# %%
