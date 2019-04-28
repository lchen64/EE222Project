import scipy
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt

N = 100 # number of iterations
N_arr = np.arange(N+1)

# function f
def f(x):
    if x < -3:
        return 8 * pow(x, 2) + 45 * x + 67.5
    elif x <= 0 and x >= -3:
        return 0.5 * pow(x, 2)
    else:
        return 8 * pow(x, 2)

# gradient of f, in this case, f'
def gf(x):
    if x < -3:
        return 16 * x + 45
    elif x <= 0 and x >= -3:
        return x
    else:
        return 16 * x

L = 16.0
m = 1.0
kappa = L / m 

def worst_func_values(k, init_dist):
    return 4 * L / pow(k+2, 2) * pow(init_dist, 2)

def worst_dist_to_optimizer(k, init_dist):
    return pow((kappa-1)/(kappa+1), k) * init_dist

#########################################################
# Nesterov's method with stepsize 1/16 and beta =3/5
# starting point x0 = 2
# x_{k+1} = x_k - alpha * f'(x_k + beta(x_k - x_{k-1})) +
#           beta(x_k - x_{k-1})
#########################################################
x0 = 2
alpha = 1.0 / 16
beta = 3.0 / 5

def run_nesterov_and_plot(alpha, beta, plot_worst_case=False):
    nesterov_values = []
    nesterov_values.append(f(x0))
    worst_values = []
    worst_values.append(worst_func_values(0, x0))
    worst_dist = []
    worst_dist.append(worst_dist_to_optimizer(0, x0))

    x_curr = x0
    x_prev = x0
    for i in range(N):
        x_next = x_curr - alpha * gf(x_curr + beta * (x_curr - x_prev)) 
        x_next += beta * (x_curr - x_prev)
        f_x = f(x_next)
        nesterov_values.append(f_x)
        x_prev = x_curr
        x_curr = x_next
        if plot_worst_case:
            worst_values.append(worst_func_values(i+1, x0))
            worst_dist.append(worst_dist_to_optimizer(i+1, x0))


    # Plot function value versus iteration number
    nesterov_values = np.array(nesterov_values)
    plt.figure(1)
    plt.plot(N_arr, nesterov_values)
    plt.xlabel("iteration number")
    plt.ylabel("function value")
    plt.title("Nesterov's method")
    plt.show()
    
    if plot_worst_case:
        worst_values = np.array(worst_values)
        plt.figure(2)
        plt.plot(N_arr, worst_values)
        plt.xlabel("iteration number")
        plt.ylabel("Upper bound on function value")
        plt.title("Nesterov's method")
        plt.show()
        
        worst_dist = np.array(worst_dist)
        plt.figure(3)
        plt.plot(N_arr, worst_dist)
        plt.xlabel("iteration number")
        plt.ylabel("Upper bound on distance to optimizer")
        plt.title("Nesterov's method")
        plt.show()
    
def run_nesterov(a, b, epsilon=pow(10, -3)):
    x_curr = x0
    x_prev = x0
    f_x = f(x0)
    N = 1
    while f_x > epsilon:
        x_next = x_curr - a * gf(x_curr + b * (x_curr - x_prev)) 
        x_next += b * (x_curr - x_prev)
        f_x = f(x_next)
        x_prev = x_curr
        x_curr = x_next
        N += 1
    return N

# Running Nesterov's method with alpha=1/16 and beta=3/5 and plot
run_nesterov_and_plot(alpha, beta, True)

# Find parameters that achieves fastest rate of convergence
alpha = np.linspace(0.01, 1, 100)
beta = np.linspace(0.01, 1, 100)
epsilon = pow(10, -4)
min_param = None
min_num_iter = float("infinity")
for a in alpha:
    for b in beta:
        # n = run_nesterov(a, b)
        if n < min_num_iter:
            min_num_iter = n
            min_param = (a, b)
print("==== Param to achieve fastest convergence: (a,b) = {}".format(min_param))

# Running Nesterov's method with best param
print("Running Nesterov with the best parameters found (a, b) = {}".format(min_param))
run_nesterov_and_plot(min_param[0], min_param[1])


