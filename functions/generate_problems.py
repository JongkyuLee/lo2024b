import numpy as np
from scipy.optimize import linprog

def genLO(m, n, mu):
    while True:
        A = np.random.random((m, n))
        int_x = np.random.rand(n)
        int_s = mu / int_x
        b = A @ int_x
        int_y = np.ones(m) - np.random.rand(m)
        c = A.T @ int_y + int_s
        int_y = np.linalg.pinv(A.T) @ (c - int_s)

        bound = [0, None]
        bounds = []
        for i in range(n):
            bounds.append(bound)

        res = linprog(c, A_eq=A, b_eq=b, bounds=bounds)
        if res.message != "The algorithm terminated successfully and determined that the problem is infeasible.":
            if res.message != "The algorithm terminated successfully and determined that the problem is unbounded.":
                if res.message != "The problem is unbounded. (HiGHS Status 10: model_status is Unbounded; primal_status is At upper bound)":
                    solx = res['x']
                    print(f"optimal value = {np.sum(c * solx)}")
                    print(f"c value = {c}" )
                    return int_x, int_y, int_s, A, b, c

def bouafia_prob(m):
    """
    Bouaafia, D.Benterki and Y.Adnan, An efficient primal-dual interior point method for linear programming problems
    based on a new kernel function with a trigonometric barrier term, J.Optim.Theory Appl., 170(2016), 528–545.
    """

    n = 2 * m
    A = np.zeros((m, n))
    for i in range(m):
        A[i, i] = 1
        if i + m < n:
            A[i, i + m] = 1
    c = np.array([-1] * n)
    for i in range(n):
        if i + m < n:
            c[i + m] = 0
    b = np.array([2] * n)

    int_x = np.array([1] * n )
    int_y = np.array([-2] * m )
    int_s = np.array([1] * n )
    int_s[m:] = 2
    return int_x, int_y, int_s, A, b, c
