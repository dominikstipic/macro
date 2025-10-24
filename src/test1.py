import matplotlib.pyplot as plt
from dp import UtilityFactory, DP
import pandas as pd
import numpy as np

capital = np.array([0.5, 1.0, 1.5])

alpha = 0.3
delta = 1
sigma = 3
beta = 0.95

def utility(alpha):
        return lambda k1, k2: np.log(k1**alpha - k2)



dp = DP(states=capital, 
        utility_function=utility(alpha), 
        alpha=alpha, 
        beta=beta, 
        capital_deprec=delta, 
        epsilon=0.0001)

values, policy = dp.run(2)
print(values)
print(policy)
