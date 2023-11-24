import random

def simulate_game():
    # probabilitatle pentru fiecare jucator
    prob_J0 = 1/2  # J0 are o monedă normala
    prob_J1 = 2/3  # J1 are o monedă masluita

    # determin cine începe jocul (J0 sau J1)
    starter = random.choice(["J0", "J1"])

    # Jucatorul desemnat arunca cu propria moneda
    n = sum([random.random() < (prob_J0 if starter == "J0" else prob_J1) for _ in range(1)])

    # Celalalt jucator aruncă cu moneda proprie de n + 1 ori
    m = sum([random.random() < (prob_J1 if starter == "J0" else prob_J0) for _ in range(n + 1)])

    return starter if n >= m else ("J1" if starter == "J0" else "J0")

# simulez 10000 jocuri
win_counts = {"J0": 0, "J1": 0}
for _ in range(10000):
    winner = simulate_game()
    win_counts[winner] += 1

print("Rezultatul simularii este :")
print(win_counts)



from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# N = nr de steme obtinute în prima runda
# M = nr de steme obtinute in a doua runda
# Starter = jucatorul care incepe : J0 sau J1
model = BayesianModel([('Starter', 'N'), ('Starter', 'M'), ('N', 'M')])


cpd_starter = TabularCPD(variable='Starter', variable_card=2, values=[[0.5], [0.5]], 
                         state_names={'Starter': ["J0", "J1"]})
cpd_N = TabularCPD(variable='N', variable_card=2, evidence=['Starter'], evidence_card=[2],
                   values=[[0.5, 1/3], [0.5, 2/3]], 
                   state_names={'Starter': ["J0", "J1"], 'N': [0, 1]})
cpd_M = TabularCPD(variable='M', variable_card=3, evidence=['N', 'Starter'], evidence_card=[2, 2],
                   values=[[1, 0.5, 1/3, 0], [0, 0.5, 1/3, 1/3], [0, 0, 1/3, 2/3]],
                   state_names={'N': [0, 1], 'Starter': ["J0", "J1"], 'M': [0, 1, 2]})


model.add_cpds(cpd_starter, cpd_N, cpd_M)


if model.check_model():
    print("Modelul este corect definit")

# interogarea modelului
inference = VariableElimination(model)
prob_starter = inference.query(variables=['Starter'], evidence={'M': 1})
print(prob_starter)

"""
OUTPUT :

Rezultatul simularii este :
{'J0': 3882, 'J1': 6118}
Modelul este corect definit
+-------------+----------------+
| Starter     |   phi(Starter) |
+=============+================+
| Starter(J0) |         0.3000 |
+-------------+----------------+
| Starter(J1) |         0.7000 |
+-------------+----------------+

"""