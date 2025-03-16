# Three versions of the Neoclassical Growth Model.

- `NGM_VFI_gridsearch.jl': solves NGM with VFI, using discrete optimization.
- `NGM_VFI.jl': solves NGM with VFI, using continuous optimization.
- `SNGM_VFI.jl': solves Stochastic Neoclassical Growth Model (SNGM) with VFI, using continuous optimization.

If markets are perfectly competitive and if other technical conditions are met (see Stokey, Lucas and Prescott, Chapter 2), the First Fundamental Theorem of Welfare Economics (FWT) characterizes the decentralized competitive equilibrium allocations as Pareto optimal, such that they are also the solutions to the problem of a benevolent planner that directly chooses allocations in a centralized fashion. 

```math
\max_{\left\lbrace c_{t}, k_{t+1}\right\rbrace_{t=0}^{\infty}}  \quad \sum^{\infty}_{t=0} \beta^t u\left(c_{t}\right) \quad s.t. \quad c_{t} + k_{t+1} \leq f(k_{t}) + \left(1-\delta\right)k_{t} \quad \forall t, \quad c_{t},k_{t+1}\geq 0 \quad \forall t.
```
Assume the utility function satisfies the usual properties of monotonicity and concavity and the Inada conditions. By virtue of the first assumption, the period budget constraint binds with equality, such that solving it out for $c_{t}$ and substituting for $c_{t}$ inside the utility function, one can formulate the planner problem above in recursive form by means of the following Bellman equation. 

```math
V\left(k\right) = \max_{k'\in \Gamma\left(k\right)} \quad \left\lbrace u\left(f(k) + \left(1-\delta\right)k - k'\right) + \beta V(k') \right\rbrace \quad s.t. \quad 
\Gamma\left(k\right) = \left[0,f(k) + \left(1-\delta\right)k\right], \forall k \in \mathbb{R}_{+}
```
where a prime denotes the value of a variable at the beginning of the period t+1. 
The Bellman equation is a functional equation in $V(\cdot)$, whose root is $V^{*}(\cdot)$.
One possible solution method, is *value function iteration* (VFI), which relies on the contraction mapping property of the Bellman operator in the space of bounded and continuous functions (see the Contraction Mapping Theorem). 
The Bellman operator T provides a rule that maps the value function $V(\cdot)$ on the right-hand side to the value function $TV(\cdot)$ on the left-hand side of the following equation, and the contraction mapping theorem ensures that such mapping has a unique fixed point, stipulating that the sequence of value functions $\left\lbrace V^{n}(\cdot) \right\rbrace^{N}_{n=1}$ resulting from iterated application of this mapping, $TV^{n}(\cdot) = V^{n-1}(\cdot)$, converges to $V^{*}(\cdot)$.
To see this clearly, let the Bellman operator T be defined such that the following relationship holds 

```math
TV\left(k\right) = \max_{k'\in \left[0,f(k) + \left(1-\delta\right)k\right]} \quad \left\lbrace u\left(f(k) + \left(1-\delta\right)k - k'\right) + \beta V(k') \right\rbrace.
```
Hence, at the optimum, the maximum value function $V^{*}(\cdot)$ (i.e. the solution to the Bellman equation) is nothing else than the fixed point of the Bellman operator T, that is 

```math
T V^{*}(\cdot) = V^{*}(\cdot).
```

The reader will appreciate how the Bellman operator provides a convenient transformation of the problem from a difficult root-finding problem to fixed-point problem for which, under conditions that are respected in the setting at hand, we know convenient convergence theorems.
