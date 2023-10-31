import pandas
import pymc3 as pm

data = pandas.read_csv("trafic.csv")
cars = data["nr. masini"].values
mins = data["minut"].values

interv_distr = list()

with pm.Model() as model:
    lmbd = pm.Normal("lambda", mu=0, sigma=10)
    traffic_total = pm.Poisson('traffic', mu=lmbd, observed=data)
    interv_distr.append(pm.Poisson(f'lambda_1', mu=lmbd , observed = cars [ (mins >= 0) and (mins < 3*60)] ) )
    interv_distr.append(pm.Poisson(f'lambda_2', mu=lmbd * 1.3 , observed= cars [ ( mins >= 3*60) and (mins < 4*60) ] ) )
    interv_distr.append(pm.Poisson(f'lambda_3', mu=lmbd * 0.4 , observed= cars [ (mins >= 4*60) and (mins < 12*60)] ) )
    interv_distr.append(pm.Poisson(f'lambda_4', mu=lmbd * 1.5 , observed= cars [ (mins >= 12*60) and (mins < 15*60) ] ) )
    interv_distr.append(pm.Poisson(f'lambda_5', mu=lmbd * 0.5 , observed= cars [ (mins >= 15*60) and (mins < 20*60) ] ) )

with model:
    trace = pm.sample(10000,tune=5000  , step=pm.Metropolis())

pm.plot_trace(trace)