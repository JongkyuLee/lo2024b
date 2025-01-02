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
In this paper, we proposed new interior-point methods (IPMs) for linear optimization (LO) problem based on a generalized class of kernel functions, originally defined by [1]. 
We improved complexity by a constant factor over the original method. 
New search directions and proximity measures are defined based on this kernel function. 
We prove that the complexity is $\mathcal{O}(\sqrt{n}(\log(n))\log(\frac{n\mu^0}{\epsilon}))$ for large-update methods and $\mathcal{O}(\sqrt{n}\log(\frac{n\mu^0}{\epsilon}))$ for small-update methods, where $n$ is a dimension of the problem, $\mu^0 > 0$, and $\epsilon > 0$. 
These represent the theoretically best complexity results for such methods so far. 
We have also demonstrated better results compared to interior-point algorithms using other kernel functions in numerical tests.

[1] Y.Y. Cho and G.M. Cho, New interior-point methods for $P_*{\kappa}$-nonlinear complementarity problems, 
J. Nonlinear Convex Anal., 22, 901-917 (2021)
