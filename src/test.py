import matplotlib.pyplot as plt
import numpy as np
from dp import UtilityFactory, value_iteration

# Check the transformation from state index matrix to the capital level matrix with vectorize

alpha = 0.3
delta = 1
sigma = 0.6
beta = 0.9

capital = [0.1, 0.5, 1.0, 1.5, 2.0]
u = UtilityFactory.utility1(alpha=alpha, sigma=sigma, delta=delta)
state_values, state_path = value_iteration(states=capital, utility_function=u, alpha=alpha, beta=beta, capital_deprec=delta, max_time=20)
print(state_path)