import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az 

U = 30  # medie 30 minute
O = 10  # deviatie standard 10 minute

# generez 100 de valori conform distributiei normale
timpi_asteptare = np.random.normal(U, O, 100)

print(timpi_asteptare)

"""
[36.11059575 35.15919366 37.4510196  48.49472844 39.651327   29.30764075
 31.07580782 46.17177165 30.83841769 39.65508312 19.26721217 25.81172333
 21.71543845 21.20932153 48.15563383 34.62669741 14.59875554 32.24453687
 28.16807787 36.17569544 44.63129441 20.93752371 13.98595531 26.81917489
 28.29357861 15.85950111 27.13571539 24.13341673 12.54179516 32.87039248
 55.70800332 31.89194506 33.92658925 31.75780638 28.75411304 57.55776428
 19.38016712 26.59445969 36.39438547 32.70233809 34.16794279 24.0142801
 44.68506423 42.73200205 29.2698574  17.17150628 42.11382746 39.25107902
 34.18530334 35.65092352 34.512559   20.12485395 34.50032992 25.04171584
 29.18345375 28.06909415 36.1086528  23.42878233 36.91542028 49.24847515
 17.78661052 29.66489732 22.57584865 29.68458364 23.02780582 14.02396429
 21.48377137 20.66822337 47.74963138 15.49748993 38.6581637  25.13075727
 25.24381559 27.94394027 10.80446409 29.56956338 46.11091548 20.35913932
 38.33514036 16.13128545 10.50692244 17.70257714 14.71534851 31.55963074
 25.83892911 41.39480226 24.22584075 39.20486684 37.22951649 30.47786318
 36.84560595 42.78481973 21.87370649 41.53772051 28.69528703 28.27303891
 41.13197702 32.44399972 33.22457005 37.72287256]
"""


with pm.Model() as model:
    # distributia a priori pentru U (medie)
    mu = pm.Normal('mu', mu=30, sigma=10)
    # distributia a priori pentru O (deviație standard)
    sigma = pm.Uniform('sigma', lower=0, upper=20)

    # modelul de observații
    observatii = pm.Normal('observatii', mu=mu, sigma=sigma, observed=timpi_asteptare)

    # rulez inferenta Bayesiana
    trace = pm.sample(1000, return_inferencedata=True)

# vizualizez rezultatele
az.plot_trace(trace)
plt.show()

# justificarea alegerii distributiilor a priori
justificare = """
Alegerea distribuției normale pentru U se bazeaza pe presupunerea ca timpl mediu de asteptare la coadă 
într o bancă urmează o distribuție normală în jurul unei medii specifice (30 de minute în acest caz).

Pt. deviațta standard O, alegem o distribuție uniformă deoarece avem o incertitudine mai mare cu privire
la valorile pe care le poate lua acest parametru, considerand că variază între 0 și 20 de minute.
"""