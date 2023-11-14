import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np

df = pd.read_csv('auto-mpg.csv')

sns.scatterplot(x='horsepower', y='mpg', data=df)
plt.title('Relatia dintre cai putere si mile pe galon')
plt.xlabel('Cai putere (CP)')
plt.ylabel('Mile pe galon (mpg)')
plt.show()

with pm.Model() as mpg_model:
    
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)

    
    mu = alpha + beta * df['horsepower']

    mpg = pm.Normal('mpg', mu=mu, sd=10, observed=df['mpg'])

with mpg_model:
    trace = pm.sample(1000, tune=1000)

print(pm.summary(trace).round(2))

sns.scatterplot(x='horsepower', y='mpg', data=df)
pm.plot_posterior_predictive_glm(trace, samples=100,
                                 eval=np.linspace(df['horsepower'].min(), df['horsepower'].max(), 100), color='blue',
                                 alpha=0.1)

plt.title('Relatia dintre cai putere si mile pe galon cu HDI')
plt.xlabel('Cai putere (CP)')
plt.ylabel('Mile pe galon (mpg)')
plt.show()