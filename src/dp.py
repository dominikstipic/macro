from typing import List, Dict, Callable, Tuple

import numpy as np
import pandas as pd


def production(capital: np.array, alpha: float, augmentation=1) -> np.array:
    """
        Production function with augmentation. 
    """
    return augmentation*(capital**alpha)

def utility(k_now, k_next, alpha, sigma, delta):
    """
        Utility function with numpy
    """
    ct = production(k_now, alpha) + (1-delta)*k_now - k_next
    #if ct == 0: return np.nan
    u = (ct**(1-sigma)-1)/(1-sigma)
    return u

def utility_log(k_now, k_next, alpha, delta):
        """
           Log utility used in case when sigma = 1. Derived with L'Hospital rule on original utility
        """
        ct = production(k_now, alpha) + (1-delta)*k_now - k_next
        return np.log(ct)

def utility_matrix(utility, capitals):
    """
        Utility for different levels of capital
    """
    n = len(capitals)
    X = np.zeros((n, n))    
    for i, c1 in enumerate(capitals):
        for j, c2 in enumerate(capitals):
            X[i, j] = utility(c1, c2)
    return X

class UtilityFactory:
    """
        Build and stores various utilities
    """
    def utility1(alpha: float, sigma: float, delta: float):
        return lambda k1, k2 : utility(k1, k2, alpha=alpha, sigma=sigma, delta=delta)

    def utility2(alpha: float, delta: float):
        """
            Sigma = 1
        """
        return lambda k1, k2 : utility(k1, k2, alpha, 1.0, delta)    
    
    def utility3(alpha: float, delta: float):
        return lambda k1, k2 : utility_log(k1, k2, alpha, delta)  


class DP:
    init_values = np.array([])
    t = 0

    def __init__(self, states, utility_function, alpha, beta, capital_deprec, epsilon=0.0001):
        """
            states: capital levels or states. 
            utility_function: function of current and future capital.
            alpha: production parameter
            beta: discount factor
            capital_deprec: capital depreciation
            epsilon: convergence and float comparison
        """
        self.states=states
        self.utility_function=utility_function
        self.alpha=alpha
        self.beta=beta
        self.capital_deprec=capital_deprec
        self.epsilon=epsilon
        self.state_values = np.zeros(len(states))
        

    def set_init(self, init_states):
        """Initialize first values of capital"""
        self.init_values = init_states

    def bellman(self, utility: Callable[[float, float], float], state_values: Dict[float, float], s0: float, states: List[float], current_capital_stock: list, beta: float) -> Tuple[float, float]:
        """
            Bellman iteration with numpy array
        """
        # 0 < k' < f(k) - Feasible set
        states[(states > current_capital_stock) | (states < 0)] = np.nan
        values = utility(s0, states) + beta*state_values
        return np.nanargmax(values), np.nanmax(values)

   

    def _next(self):
        """ 
            Does one value iteration of the DP problem
            Returns: values, optimal policy indexes, deltas between current and previous iteration
        """
        new_state_values = np.zeros(len(self.states))
        optim_policy = np.array([])
        for i, s_current in enumerate(self.states):
                current_capital_stock = production(s_current, self.alpha) + (1-self.capital_deprec)*s_current
                optim_index, optim_value = self.bellman(self.utility_function, self.state_values, s_current, np.array(self.states), current_capital_stock, self.beta)
                new_state_values[i] = optim_value
                optim_policy = np.append(optim_policy, optim_index)
        deltas = abs(new_state_values - self.state_values)
        self.t += 1
        return new_state_values, optim_policy, deltas
    
    def next(self) -> Tuple[np.array, np.array]:
        """ 
            Calls the private _next method. Discard deltas used as convergence criterion. 
            Returns: values, optimal policy indexes
        """
        new_state_values, optim_policy, _ = self._next()
        return new_state_values, optim_policy.astype(int)

    def run(self, max_time=1000) -> Tuple[np.array, np.array]:
        """ 
            Used for running DP optimization with max time parameter. 
            Returns: values, optimal policy indexes
        """
        state_path = np.arange(0, len(self.state_values)).reshape(-1, 1)
        while self.t < max_time:
            self.state_values, optim_policy, deltas = self._next()
            state_path = np.hstack([optim_policy.reshape(-1, 1), state_path])
            if (deltas < self.epsilon).all():
                print(f"Coverged in iter = {self.t}")
                break
            avg_value = self.state_values.mean()
            delta = abs(deltas - self.epsilon).max()
            print(f'iter = {self.t}, avg_value = {avg_value} delta={delta}')
        for i in range(len(state_path)): 
            state_path[i] = state_path[i][::-1]
        return self.state_values, state_path

def write_csv(ds: dict, csv_name='out.csv'):
    """
        Write to dict to csv 
    """
    ds = {k: [v] for k, v in ds.items()}
    df = pd.DataFrame(ds)
    df.to_csv(csv_name, index=False)


if __name__ == "__main__":
    alpha = 0.4
    delta = 0.04
    beta = 0.96
    epsilon = 0.0001

    capital = np.linspace(5, 20, num=1000)
    utility = UtilityFactory.utility3(alpha, delta)
    
    dp = DP(states=capital, 
            utility_function=utility, 
            alpha=alpha, 
            beta=beta, 
            capital_deprec=delta, 
            epsilon=epsilon)
    state_values, state_path = dp.run(300)
    print(state_path)
    np.save('../repo/path.npy', state_path)
    np.save('../repo/state_values.npy', state_values)


