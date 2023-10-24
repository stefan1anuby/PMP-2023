import numpy as np


lambda_poisson = 20
mu_normal = 2
sigma_normal = 0.5

def simulate(alpha, num_simulations=10000):
    total_time = []
    for _ in range(num_simulations):
        num_clients = np.random.poisson(lambda_poisson)
        order_time = np.random.normal(mu_normal, sigma_normal, num_clients)
        cooking_time = np.random.exponential(alpha, num_clients)
        total_time.append(sum(order_time + cooking_time))
    return total_time

def find_alpha(target_time=15*20, probability=0.95, num_simulations=10000):
    # incep cu o estimare initiala și ajustez alpha pana cand ajung la probabilitatea dorita
    alpha_estimate = 1
    while True:
        total_time = simulate(alpha_estimate, num_simulations)
        prob = sum([1 for t in total_time if t <= target_time]) / num_simulations
        if prob >= probability:
            break
        alpha_estimate -= 0.01
    return alpha_estimate

alpha_max = find_alpha()
waiting_time = (2 + alpha_max) / 2

print(f"Valoarea maxima a lui alpha este aproximativ: {alpha_max:.2f} minute")
print(f"Timpul mediu de așteptare pentru a fi servit este: {waiting_time:.2f} minute")
