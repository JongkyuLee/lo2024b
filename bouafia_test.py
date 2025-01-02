from functions.implementation import *

tau = 2
epsilon = 10 ** (-5)
theta = 0.5
int_mu = 1

result = []
for m in np.arange(100, 510, 10):
    print("====================================")
    tmp_result = []
    int_x, int_y, int_s, A, b, c, n = bouafia_prob(m)

    test_kernel_list = [2, 3, 4, 5, "mousaab", "benhadid21", "benhadid23", "fathi"]
    for ker_num in test_kernel_list:
        x, y, s = int_x.copy(), int_y.copy(), int_s.copy()
        mu = int_mu

        p = p_value(ker_num, n)
        q = q_value(ker_num, n, p)
        total_itr = implementation(ker_num, A, p, q, m, n, mu, x, y, s, theta, epsilon, tau)
        tmp_result.append(total_itr)

    print(f"{m} : {tmp_result}")
    result.append(tmp_result)

result = np.array(result)
print(result)

