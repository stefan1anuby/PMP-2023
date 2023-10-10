import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

numar_simulari = 10000

distributii_server = [
    stats.gamma(4, scale=1/3),
    stats.gamma(4, scale=1/2),
    stats.gamma(5, scale=1/2),
    stats.gamma(5, scale=1/3)
]

probabilitati_server = [0.25, 0.25, 0.30, 0.20]

distributie_latenta = stats.expon(scale=1/4)

timpuri_raspuns = np.zeros(numar_simulari)

for i in range(numar_simulari):
    
    server_ales = np.random.choice(4, p=probabilitati_server)
    
    timp_raspuns_server = distributii_server[server_ales].rvs()
    timp_latenta = distributie_latenta.rvs()
    
    timpuri_raspuns[i] = timp_raspuns_server + timp_latenta

probabilitate = np.mean(timpuri_raspuns > 3)

print(f"Probabilitatea ca timpul de raspuns sa fie mai mare de 3 ms este {probabilitate:.4f}")

az.plot_posterior(timpuri_raspuns)
plt.show()
