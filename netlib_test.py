from functions.implementation import *
import os
import time

directory = 'netlib_mat/'
sorted_files = sorted(os.listdir(directory))
tau = 2
epsilon = 10 ** (-5)
theta = 0.5
int_mu = 1

result = []
result_time = []

for filename in sorted_files:
    print("====================================")
    print(filename)
    file_path = os.path.join(directory, filename)
    int_x, int_y, int_s, A, b, c, m, n = netlib(file_path, int_mu)

    tmp_result = []
    tmp_time_result = []

    test_kernel_list = [2, 3, 4, 5, "mousaab", "benhadid21", "benhadid23", "fathi"]
    for ker_num in test_kernel_list:
        x, y, s = int_x.copy(), int_y.copy(), int_s.copy()
        mu = int_mu

        p = p_value(ker_num, n)
        q = q_value(ker_num, n, p)
        start_time = time.time()
        total_itr = implementation(ker_num, A, p, q, m, n, mu, x, y, s, theta, epsilon, tau)
        elapsed_time = time.time() - start_time

        tmp_result.append(total_itr)
        tmp_time_result.append(round(elapsed_time, 4))

    print(f"{m} : {tmp_result}")
    print(f"{m} : {tmp_time_result}")

    result.append(tmp_result)
    result_time.append(tmp_time_result)

result = np.array(result)
print(result)

print("Final Time Results (seconds):")
print(result_time)


