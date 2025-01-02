import numpy as np
from scipy.integrate import quad

def fun_m(p,q):
    tmp = (q + 2)**2 / (np.pi * p * q * ( 1/np.tan((np.pi)/(q+2)) + np.tan((np.pi)/(q+2)) ))
    return tmp

def u(q,t):
    return (np.pi)/(q*t + 2)

def up(q,t):
    return -((np.pi * q)/((q*t + 2)**2))

def k(p,q):
    return np.tan(np.pi/(q+2))**(p-1) + np.tan(np.pi/(q+2))**(p+1)

def integrand_eligible_5(t):
    return np.exp(1)**((1/t)-1)



def kernel(ker_num, t, p, q):
    # our new kernel
    if ker_num in [0, 1, 2]:
        tmp = (t**2-1)/2 - (1-3/(2*q))*t + 1/(2*q*t) \
          + 1/(p*q*(t**p)) + 1 - 2/q - 1/(p*q)

    # eligible kernel
    elif ker_num == 3:
        tmp = (t ** 2 - 1) / 2 - np.log(t)

    elif ker_num == 4:
        tmp = 0.5 * (t - 1 / t) ** 2

    elif ker_num == 5:
        tmp = (t ** 2 - 1) / 2 + (t**(1-q) - 1)/(q - 1)

    elif ker_num == "mousaab": # p >= 2, q > 0
        p = 2
        q = 1
        tmp = (t**2 - 1)/2 + fun_m(p,q) * ((1/np.tan((np.pi)/(q+2))**p ) * (np.tan(u(q,t)))**p - 1)

    elif ker_num == "benhadid21": # p is natural number, q > 1
        tmp_sum = 0
        for j in range(1, p+1):
            tmp_sum += (t**(1 - q**j) -1)/(1 - q**j)
        tmp = (t**2 -1)/2 - (1/p) * tmp_sum

    elif ker_num == "benhadid23": # p > 1
        tmp = (t**2 -1) - (t**(-2*p + 1) - 1)/(-2*p +1) - (t**(-p +1) - 1)/(-p +1)

    elif ker_num == "fathi":
        def h_fathi(t):
            return (np.pi * t) / (2 + 2 * t)
        def integrand_fathi(p, t):
            return np.exp(p * (1 - np.tan(h_fathi(t))))
        
        growth = (t ** 2 - 1) / 2
        barrier = []
        for _ in range(growth.shape[0]):
            barrier.append(quad(integrand_fathi, 1, t[_], args=(p,))[0])
        tmp = growth - np.array(barrier)

    return np.sum(tmp)



def der_kernel(ker_num, t, p, q):
    global tmp
    # our new kernel
    if ker_num in [0, 1, 2]:
        tmp = t - (1 - 3 / (2 * q)) - 1 / (2 * q * (t ** 2)) - 1 / (q * (t ** (p + 1)))

    # eligible kernel
    elif ker_num == 3:
        tmp = t - 1/t
    elif ker_num == 4:
        tmp = t - 1/(t**3)
    elif ker_num == 5:
        tmp = t - 1/(t**(q))

    elif ker_num == "mousaab": # p >= 2, q > 0
        tmp_1 = p * fun_m(p,q) * (1/np.tan(np.pi/(q+2))**p)
        tmp_2 = (1 + np.tan(u(q,t))**2 ) * (np.tan(u(q,t))**(p-1)) * up(q,t)
        tmp = t + tmp_1 * tmp_2

    elif ker_num == "benhadid21": # p is natural number, q > 1
        tmp_sum = 0
        for j in range(1, p+1):
            tmp_sum += t**(-q**j)
        tmp = t - (1/p) * tmp_sum

    elif ker_num == "benhadid23": # p > 1
        tmp = 2*t - t**(-2*p) - t**(-p)

    elif ker_num == "fathi": # p >= 3
        def h_fathi(t):
            return (np.pi * t) / (2 + 2 * t)

        tmp = t - np.exp(p * (1 - np.tan(h_fathi(t))))
    return tmp

def stepsize(ker_num, delta, p, q):
    if ker_num in [0, 1, 2]:
        tmp = (np.sqrt(2) + (1/q) * (1+4*q)**(3/(p+1))
            + ((p+1)/q)*(1+4*q)**((p+2)/(p+1)))
        tmp_res = tmp * delta**((p+2)/(p+1))

    elif ker_num == 3:
        tmp_res = 19 * (delta**2)
    elif ker_num == 4:
        tmp_res = 27 * (delta**(4/3))
    elif ker_num == 5:
        tmp_res = 1 + q*((1+4*delta)**((q+1)/q))

    elif ker_num == "mousaab": # p >= 2, q > 0
        tmp_1 = p * fun_m(p,q) * (1/(np.tan(np.pi/(q+2))**p )) * ( 1 + (k(p,q))**(2/(p+1)) )
        tmp_2 = ((p-1) * (k(p,q))**((p-2)/(p+1)) + (p+1) * (k(p,q))**(p/(p+1)) ) * (((q*np.pi)/4)**2)
        tmp_3 = (k(p,q))**((p-1)/(p+1)) * np.pi * (q/2)**2
        tmp_res = ( (4*delta + 1)**((p+2)/(p+1)) ) * (1 + tmp_1 * (tmp_2 + tmp_3))

    elif ker_num == "benhadid21": # p is natural number, q > 1
        tmp_sum = 0
        for j in range(1, p+1):
            tmp_sum += (q**j)* ( (4 * p * delta + p)**( (q**j+1)/(q**p)) )
        tmp_res = 1 + (1/p) * tmp_sum

    elif ker_num == "benhadid23": # p > 1
        tmp_res = 2 + 3*p*(4*delta + 1)**((2*p + 1)/(2*p))

    elif ker_num == "fathi": # p >= 3
        tmp_res = (14 * np.pi * p * delta)

    return (1 / tmp_res)

def p_value(ker_num, n):
    if ker_num in [0, 1, 2]:
        p = 0.5 * np.log(n) - 1
        if p < 1:
            p = 1
    # eligible kernel function
    elif ker_num in [3, 4, 5]:
        p = 0
    elif ker_num == "mousaab":  # p >= 2, q > 0
        p = 2
    elif ker_num == "boukhenchouche":
        p = 1
    elif ker_num == "benhadid21":  # p is natural number, q > 1
        p = 1
    elif ker_num == "benhadid23":  # p > 1
        p = 2
    elif ker_num == "fathi":  # p >= 3
        p = 3
    return p

def q_value(ker_num, n, p):
    if ker_num == 0:
        q = 3 / 2
    elif ker_num == 1:
        q = 3
    elif ker_num == 2:
        q = p + 2
    elif ker_num == 3:
        q = 3
    elif ker_num == 4:
        q = 3
    elif ker_num == 5:
        q = 0.5 * np.log(n)

    elif ker_num == "mousaab": # p >= 2, q > 0
        q = 1
    elif ker_num == "benhadid21": # q > 1
        q = 2
    elif ker_num == "benhadid23": # no q
        q = 1
    elif ker_num == "fathi": # no q
        q = 1

    return q