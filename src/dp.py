from typing import List, Dict, Callable, Tuple

import numpy as np
import pandas as pd

def utility(k_now, k_next, alpha, sigma, delta):
        ct = production(k_now, alpha) + (1-delta)*k_now - k_next
        if ct == 0: return np.nan
        u = (ct**(1-sigma)-1)/(1-sigma)
        print(f"k={k_now}, k_future={k_next}, ct={ct}, ut={u}")

        return u

def utility_log(k_now, k_next, alpha, delta):
        ct = production(k_now, alpha) + (1-delta)*k_now - k_next
        return np.log(ct)

def utility_matrix(utility, capitals):
    n = len(capitals)
    X = np.zeros((n, n))    
    for i, c1 in enumerate(capitals):
        for j, c2 in enumerate(capitals):
            X[i, j] = utility(c1, c2)
    return X

class UtilityFactory:
    def utility1(alpha: float, sigma: float, delta: float):
        return lambda k1, k2 : utility(k1, k2, alpha=alpha, sigma=sigma, delta=delta)

    def utility2(alpha: float, delta: float):
        """
            Sigma = 1
        """
        return lambda k1, k2 : utility(k1, k2, alpha, 1.0, delta)    
    
    def utility3(alpha: float, delta: float):
        return lambda k1, k2 : utility_log(k1, k2, alpha, delta)  

def production(capital, alpha):
    return capital**alpha

def consumption(k_now, k_next, alpha, sigma, delta):
    x = (k_now**alpha) + (1-delta)*k_now - k_next
    x = x**(1-sigma)
    return (x-1)/(1-sigma)

def bellman(utility: Callable[[float, float], float], state_values: Dict[float, float], s0: float, states: List[float], current_capital_stock: list, beta: float) -> Tuple[float, float]:
    state_value_list = [(s, utility(s0, s) + beta*state_values[i]) for i, s in enumerate(states) if s <= current_capital_stock]
    if s0 == 2.0:
        print()
    state_value = max(state_value_list, key=lambda x : x[1])
    return state_value[0], state_value[1]

def index_of(state, states, precision=0.0001):
    for i, s in enumerate(states):
        if abs(s - state) <= precision:
            return i
    return -1

def value_iteration(states, utility_function, alpha, beta, capital_deprec, epsilon=0.0001, max_time=10000):
    state_values = [0 for _ in range(len(states))]
    state_path = [[] for _ in states]
    for t in range(max_time):
        deltas = np.array([])
        # Prevent modifications of old values while calucating in the value iteration
        new_state_values = np.zeros(len(states))
        new_state_path = list(state_path)
        for i, s_current in enumerate(states):
            # For the current node s_current, find the future node with the optimal value. 
            # value = max { reward(s_current, s_future) + beta * V(s_future) }
            current_capital_stock = production(s_current, alpha) + (1-capital_deprec)*s_current
            optim_state, optim_value = bellman(utility_function, state_values, s_current, states, current_capital_stock, beta)
            delta = optim_value - state_values[i]
            deltas = np.append(deltas, delta)
            new_state_values[i] = optim_value
            
            optim_index = index_of(optim_state, states)
            optim_path = state_path[optim_index]
            new_state_path[i] = [optim_index] + optim_path
        if (deltas < epsilon).all():
            print(f'Converged with time horizont T = {t} !')
            break
        # Modify old values with new one. 
        state_values = new_state_values
        state_path = new_state_path
        max_capital = max(state_values)
        avg_value = sum(state_values)/len(state_values)
        delta = min(abs(deltas - epsilon))
        print(f'iter = {t+1}, capital = {max_capital}, avg_value = {avg_value} delta={delta}')
    state_path = np.array(state_path)
    state_values = np.array(state_values)
    return state_values, state_path
    

def write_csv(ds: dict, csv_name='out.csv'):
    ds = {k: [v] for k, v in ds.items()}
    df = pd.DataFrame(ds)
    df.to_csv(csv_name, index=False)


if __name__ == "__main__":
    alpha = 0.3
    delta = 1
    sigma = 2
    beta = 0.9

    capital = [0.1, 0.5, 1.0, 1.5, 2.0]
    u = UtilityFactory.utility1(alpha=alpha, sigma=sigma, delta=delta)
    state_values, state_path = value_iteration(capital, u, alpha=alpha, beta=beta, capital_deprec=delta, max_time=10)
    print(state_values)




#def value_iteration(capital, utility, )