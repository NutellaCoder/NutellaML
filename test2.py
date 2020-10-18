# Import HyperOpt Library
from hyperopt import tpe, hp, fmin

objective = lambda x: (x-3)**2 + 2


space = hp.uniform('x', -10, 10)

best = fmin(
    fn=objective, # Objective Function to optimize
    space=space, # Hyperparameter's Search Space
    algo=tpe.suggest, # Optimization algorithm
    max_evals=1000 # Number of optimization attempts
)
print(best)