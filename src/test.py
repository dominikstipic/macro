import matplotlib.pyplot as plt
import numpy as np
from dp import UtilityFactory, value_iteration, utility_matrix


alpha = 0.4
sigma = 0.5
delta = 0.04

utility = UtilityFactory.utility1(alpha=alpha, sigma=sigma, delta=delta)
capital = np.linspace(5, 6, num=3)

print(capital)
U = utility_matrix(utility, capital)
print(U)

#############

# Check the transformation from state index matrix to the capital level matrix with vectorize
alpha = 0.4
delta = 0.04
sigma = 0.5
beta = 0.96
epsilon = 0.0001

capital = np.linspace(5, 6, num=3)
u = UtilityFactory.utility1(alpha, sigma, delta)
print(f'CAPITAL LEVELS = {capital}')
state_values, state_path = value_iteration(capital, u, beta, max_time=4)

f = np.vectorize(lambda x: capital[x])
capital_paths = f(state_path)
capital_paths

