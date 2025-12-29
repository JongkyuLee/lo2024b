## lo2024b
Jongkyu Lee · You-Young Cho · Gyeong-Mi Cho*

New interior-point methods for linear optimization problems.



## Create and Activate the Conda Environment
It is recommended to run the project in a Python 3.12 environment.  
If Conda is not installed, download and install it from the [Anaconda website](https://www.anaconda.com/products/distribution) or [Miniconda website](https://docs.conda.io/en/latest/miniconda.html).  

The following commands should be executed in **Command Prompt (Windows)** or **Terminal (macOS/Linux)**.
```bash

# 1. Create a Conda virtual environment (replace 'env_name' with your preferred environment name)
conda create -n env_name python=3.12

# 2. Activate the virtual environment
conda activate env_name

# 3. Install packasges
pip install -r requirements.txt
```


## Abstract 
In this paper, we propose new interior-point methods for solving linear optimization problemS based on a generalized class of kernel functions, originally defined in [1].
New search directions and proximity measures are defined based on these kernel functions.
We prove that the complexity is $\mathcal{O}\left(\sqrt{n} ( \log n) \log ({n}/{\epsilon})\right)$
for long-step methods and $\mathcal{O}\left(\sqrt{n}\log({n}/{\epsilon})\right)$ for short-step methods,
where $n$ is the dimension of the problem and $\epsilon>0$.
These represent the theoretically best-known complexity results for such methods so far.
%We improve complexity by a constant factor over the method in \cite{ch21}.
Finally, numerical examples are given to demonstrate the efficiency of the proposed methods.
Our method achieved the fewest iterations and shortest running time in 84\% of test cases, including 36 randomly generated problems, 25 NETLIB benchmark problems [2], and 41 instances from Bouafia et al. [3].


[1] Y.Y. Cho and G.M. Cho, New interior-point methods for $P_*{\kappa}$-nonlinear complementarity problems, 
J. Nonlinear Convex Anal., 22, 901-917 (2021)

[2] Koch, T., The final NETLIB-LP results, Oper. Res. Lett., 32, 138-142 (2004)

[3] Bouafia, M., Benterki, D., Yassine, A., An efficient primal-dual interior point method for linear programming problems based on a new kernel function with a trigonometric barrier term, J. Optim. Theory Appl., 170(2), 528-545 (2016)
