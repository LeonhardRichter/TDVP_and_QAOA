# TDVP and QAOA

This is the repository associated to the bachelor thesis of Leonhard Richter.

The _Quantum Approximate Optimization Algorithm_ (_QAOA_) first introduced by Farhi et al.[^fn1] is combined with the _Time-Dependent Variational Principle for imaginary time evolution_[^fn2].
In this repository, the _TDVP_ and the _QAOA_ are implemented for simulations using the [QuTiP](https://qutip.org/index.html) library[^fn3] [^fn4].

The version of the code that is used in the thesis is found in XXXXXXX.

The main methods are found in `TDVP_and_QAOA.py`. There, all classes and functions, that are vital for the various algorithms are defined.
`MaxCut.py` defines a class for instances of the [Maximum Cut problem](https://en.wikipedia.org/wiki/Maximum_cut).
`quantum.py` defines some basic functions for convenience.
`benchmark.py` defines among others the function `bench_series` that is used in `run_benchmarks.ipynb` for generating the results presented in the thesis.
The exact instance objects that are being considered in the thesis are 'pickled' in `./instances/n4_instances.p` and in `./instances/n5_instances.p`.
Some other jupyter-notebooks collect the data processing and analysis.

[^fn1]: Farhi, E., Goldstone, J., & Gutmann, S. (2014). A quantum approximate optimization algorithm. arXiv preprint [arXiv:1411.4028](https://arxiv.org/abs/1411.4028) [quant-ph]
[^fn2]: Haegeman, J., Cirac, J. I., Osborne, T. J., Pižorn, I., Verschelde, H., & Verstraete, F. (2011). Time-dependent variational principle for quantum lattices. Physical review letters, 107(7), 070601.
[^fn3]: J. R. Johansson, P. D. Nation, and F. Nori: "QuTiP 2: A Python framework for the dynamics of open quantum systems.", Comp. Phys. Comm. 184, 1234 (2013) [DOI: [10.1016/j.cpc.2012.11.019](https://dx.doi.org/10.1016/j.cpc.2012.11.019)].
[^fn4]: J. R. Johansson, P. D. Nation, and F. Nori: "QuTiP: An open-source Python framework for the dynamics of open quantum systems.", Comp. Phys. Comm. 183, 1760–1772 (2012) [DOI: [10.1016/j.cpc.2012.02.021](https://dx.doi.org/10.1016/j.cpc.2012.02.021)].
