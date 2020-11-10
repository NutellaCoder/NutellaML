'''
    nutella만으로 importance 까지
'''

from nutellaAgent import hpo
from nutellaAgent import nu_fmin
import numpy as np
import pandas as pd

# define an objective function
def objective(args):
    val, val2 = args['hp1'], args['hp2']
    if val>val2:
        return val + val2
    else:
        return val ** 2 - val2

space = {'hp1': 1 + hpo.hp.lognormal('a', 0, 1),
         'hp2': hpo.hp.uniform('b', 1, 3)
        }

# minimize the objective over the space
trials = hpo.Trials()
best = nu_fmin(objective, space, algo=hpo.tpe.suggest, max_evals=100, trials=trials)

print("best값은")
print(best)

