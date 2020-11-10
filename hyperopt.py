# Hyperopt 튜토리얼

from nutellaAgent import space, hpo, our_tpe #, Trials
# define an objective function
def objective(args):
    case, val = args
    if case == 'case 1':
        return val
    else:
        return val ** 2

# define a search space
space = space.choice('a',
    [
        ('case 1', 1 + space.lognormal('c1', 0, 1)),
        ('case 2', space.uniform('c2', -10, 10))
    ])

# minimize the objective over the space
# trials = Trials()
best = hpo(objective, space, our_tpe.suggest, 100)

print(best)
# print(trials.results)
# -> {'a': 1, 'c2': 0.01420615366247227}
# print(space_eval(space, best))
# # -> ('case 2', 0.01420615366247227}