# import numpy as np
# import time
# import matplotlib.pyplot as plt
# import scipy
# import os
#
# from scipy.integrate import quad
# from scipy.optimize import linprog

from functions.test_datasets import *
from functions.kernel_functions import *

def prox(ker_num, p, q, x, s, mu):
    v = np.sqrt(x * s / mu)
    return np.linalg.norm(der_kernel(ker_num, v, p, q)) / 2

def backtracking(x, y, s, dx, dy, ds, alpha, rho = 0.9):
    while True:
        new_x, new_y, new_s = x + alpha * dx, y + alpha * dy, s + alpha * ds
        if np.any(new_x < 0) or np.any(new_s < 0):
            alpha *= rho
        else:
            break
    return new_x, new_y, new_s

def newton(ker_num, A, p, q, x, y, s, m, n, mu):
    v = np.sqrt(x * s / mu)
    delta = prox(ker_num, p, q, x, s, mu)
    system = np.block([
                [A, np.zeros((m, m)), np.zeros((m, n))],
                [np.zeros((n, n)), A.T, np.eye(n)],
                [np.diag(s), np.zeros((n, m)), np.diag(x)]
    ])
    residual = np.block([np.zeros(m), np.zeros(n), -mu * v * der_kernel(ker_num, v, p, q)])
    sol = np.linalg.solve(system, residual)
    dx, dy, ds = sol[: n], sol[n : n+m], sol[-n:]
    tmp_alpha = stepsize(ker_num, delta, p, q)
    x, y, s = backtracking(x, y, s, dx, dy, ds, alpha = tmp_alpha*200, rho = 0.9)
    return x, y, s

def implementation(ker_num, A, p, q, m, n, mu, x, y, s, theta, epsilon, tau):
    total_itr = 0
    while n * mu >= epsilon:
        inner_itr = 0
        mu = (1 - theta) * mu
        v = np.sqrt(x * s / mu)
        while kernel(ker_num, v, p, q) > tau:
            x, y, s = newton(ker_num, A, p, q, x, y, s, m, n, mu)
            v = np.sqrt(x * s / mu)
            inner_itr += 1
        total_itr += inner_itr
    return total_itr
