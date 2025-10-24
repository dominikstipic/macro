import matplotlib.pyplot as plt
from dp import UtilityFactory, DP
import pandas as pd

df = pd.read_csv('../repo/test.csv')
capital = (df['k']).to_numpy()

alpha = 0.4
delta = 0.1
sigma = 3
beta = 0.95
utility = UtilityFactory.utility1(alpha=alpha, sigma=sigma, delta=delta)

dp = DP(states=capital, 
        utility_function=utility, 
        alpha=alpha, 
        beta=beta, 
        capital_deprec=delta, 
        epsilon=0.0001)

for i in range(1, 6):
    col_value, col_policy = f'V{i}', f'policy{i}'
    target_values = df[col_value].to_numpy()
    target_policy = df[col_policy].to_numpy()
    dp.state_values, optim_policy_level = dp.next()
    optim_policy = capital[optim_policy_level]
    policy_bool = optim_policy == target_policy
    delta = abs(target_values - dp.state_values) < 10**-4
    # print(f'iter={i} value success:{delta.all()} policy success:{policy_bool.all()}')
