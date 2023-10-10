import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

lambda1 = 4 
lambda2 = 6

n = 10000

p1 = 0.4
p2 = 1 - p1

mecanic = np.random.choice([1, 2], p=[p1, p2], size=n)
 
timpuri_servire = np.zeros(n)
for i in range(n):
    if np.random.rand() < p1:
        timpuri_servire[i] = stats.expon.rvs(scale=1/lambda1)
    else:
        timpuri_servire[i] = stats.expon.rvs(scale=1/lambda2)

media = np.mean(timpuri_servire)
deviatie_standard = np.std(timpuri_servire)

print(f"Media timpului de servire: {media}")
print(f"Deviatia standard a timpului de servire: {deviatie_standard}")

az.plot_posterior(timpuri_servire)
plt.show() 
