import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

fig, axes = plt.subplots(len(Y_values), len(theta_values), figsize=(10, 8))
fig.suptitle('Distribuții a Posteriori pentru n')

for i, Y in enumerate(Y_values):
    for j, theta in enumerate(theta_values):
        with pm.Model() as model:
            
            n = pm.Poisson('n', mu=10)
            
            Y_obs = pm.Binomial('Y_obs', n=n, p=theta, observed=Y)
            
            trace = pm.sample(10000, return_inferencedata=False)
            
            az.plot_posterior(trace, var_names=['n'], ax=axes[i, j])
            axes[i, j].set_title(f'Y = {Y}, θ = {theta}')

plt.tight_layout()
plt.show()
