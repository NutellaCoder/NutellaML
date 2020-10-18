acc = [0.5136481523513794,
 0.6917036771774292,
 0.7838888764381409,
 0.8411852121353149,
 0.8750740885734558,
 0.8985000252723694,
 0.913777768611908,
 0.9248703718185425,
 0.9307592511177063,
 0.9395370483398438]

import nutellaAgent

nnn = nutellaAgent.Nutella()
nnn.init("run123", "123", 0)
# nnn.config(batchSize=125,epoch=30)
nnn.log(accuracy=acc)


from nutellaAgent import space, hpo, our_tpe
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
best = hpo(objective, space, algo=our_tpe.suggest, max_evals=100)

print(best)
# -> {'a': 1, 'c2': 0.01420615366247227}
# print(space_eval(space, best))
# # -> ('case 2', 0.01420615366247227}