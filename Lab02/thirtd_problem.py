import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

numar_aruncari = 10
numar_simulari = 100

pr_stema_nemasluita = 0.5
pr_stema_masluita = 0.3

rezultate = {'ss': [], 'sb': [], 'bs': [], 'bb': []}

for i in range(numar_simulari):
    
	temp_rez = {'ss': 0, 'sb': 0, 'bs': 0, 'bb': 0}

	for j in range(numar_aruncari):
		moneda1 = 's' if np.random.rand() < pr_stema_nemasluita else 'b'
		moneda2 = 's' if np.random.rand() < pr_stema_masluita else 'b'

		temp_rez[moneda1 + moneda2] += 1
	
	for combinatie in rezultate:
	    rezultate[combinatie].append(temp_rez[combinatie])

az.plot_posterior(rezultate)
plt.show()