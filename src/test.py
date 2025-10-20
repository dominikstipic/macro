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