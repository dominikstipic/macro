import numpy as np
import matplotlib.pyplot as plt

def production(k, alpha):
    return k**alpha

def inv_product(marginal_product, alpha):
    return np.pow(marginal_product/alpha, 1/(alpha-1))

def marginal_production(k, alpha):
    return np.pow(alpha*k, alpha-1)

def utility(c, gamma):
    return np.pow(c, (1-gamma))/(1-gamma)

def marginal_utility(c, gamma, epsilon=10**-5):
    return (c+epsilon)**(-gamma)

def capital_steady_state(alpha, beta, delta):
    ro = (1/beta)-1
    return np.pow((ro+delta)/alpha, 1/(alpha-1))

def next_capital(produced, delta, capital, consumption, gov_spending):
    return produced+(1-delta)*capital-consumption-gov_spending

def gross_return(delta, marginal_product):
    return 1 - delta + marginal_product

def shoot(capital, consumption, gov_spending, alpha, beta, delta, gamma):
    capital_next = capital**alpha + (1-delta)*capital - consumption - gov_spending
    if capital_next < 0: capital_next = 0.0001
    consumption_next = consumption * ((1 + alpha * capital_next**(alpha - 1) - delta) * beta)**(1/gamma)
    return_t = 1 - delta + np.pow(alpha*capital_next, alpha-1)
    return consumption_next, capital_next, return_t

def generate_sequence(TS, c0, k0, government_spending, alpha, beta, delta, gamma, epsilon=10**-1, CDELTA=0.00001, maxiter=10000):
    kss = capital_steady_state(alpha, beta, delta)
    for _ in range(maxiter) :
        error = False
        ct = np.array([c0])
        kt = np.array([k0])
        r0 = 1 - delta + marginal_production(k0, alpha)
        rt = np.array([r0])
        for _ in range(TS):
            c, k, r = shoot(kt[-1], ct[-1], government_spending, alpha, beta, delta, gamma)
            if np.isnan(c) or np.isnan(k) or np.isnan(r) or np.isinf(c) or np.isinf(k) or np.isinf(r):
                if kt[-1] < kss: 
                    error = -np.inf # Signal collapse/undershoot
                    break
                else: 
                    error = np.inf # Signal explosion/overshoot
                    break
            ct = np.append(ct, c)
            kt = np.append(kt, k)
            rt = np.append(rt, r)

        if error == np.inf:
            c0 += CDELTA * 10
        elif error == -np.inf:
            c0 -= CDELTA * 10
        else:
            distance = (kt[-1]-kss)/kss 
            print(f'c={ct[-1]}, k={kt[-1]}, r={rt[-1]}, distance={distance}')
            if (abs(distance) <= epsilon): 
                print('SOLUTION FOUND!')
                return ct, kt, rt
            else:
                if kt[-1] > kss:
                    c0 += CDELTA
                else:
                    c0 -= CDELTA

alpha = 0.33
delta = 0.2
gamma = 2
beta = 0.95
government_spending_old = 0.2 
government_spending_new = 0.25
TS = 1000

kss = capital_steady_state(alpha, beta, delta) # 1.48
c0 = production(kss, alpha) - delta*kss - government_spending_old # 0.643

print(f'CAPITAL STEADY STATE = {kss}, CONSUMPTION STEADY STATE = {c0}')
css, kss, rss = generate_sequence(TS, c0, kss, government_spending_new, alpha, beta, delta, gamma)

def calculate_taxes(beta, government_spending, css):
    TS = len(css)
    betas = np.array([beta**i for i in range(TS)])
    S = sum(betas*css)
    return government_spending/((1-beta)*S)

t = calculate_taxes(beta, government_spending_new, css)
print(f'TAX = {t}')

