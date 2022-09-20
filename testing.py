#%%
from qaoa_and_tdvp import *
from MaxCut import *

p = 2
# qubo = np.array(
#     [
#         [0, 2.0, 1.0, 1.0, 0.0, 0.0],
#         [1, 1.0, 2.0, 0.0, 1.0, 0.0],
#         [2, 1.0, 0.0, 3.0, 1.0, 1.0],
#         [3, 0.0, 1.0, 1.0, 3.0, 1.0],
#         [4, 0.0, 0.0, 1.0, 1.0, 2.0],
#     ]
# )

instance = MaxCut(nx.triangular_lattice_graph(1, 1))
nx.draw(instance.graph, with_labels=True)

qaoa = QAOA(qubo=instance.qubo, p=p)
delta = tuple(1 for _ in range(2 * p))

# gram = qaoa.gram(delta)
#%%
res = tdvp_optimize_qaoa(
    qaoa, delta, 2, int_mode="RK45", rhs_mode="qaoa", grad_tol=0.05, max_iter=300
)
