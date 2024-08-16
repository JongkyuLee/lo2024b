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