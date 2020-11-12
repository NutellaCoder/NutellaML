'''
    nutella로 hyperopt 튜토리얼 && trials 분석을 위함
'''

from nutellaAgent import hpo
from nutellaAgent import nu_fmin

# define an objective function
def objective(args):
    case, val = args
    if case == 'case 1':
        return val
    else:
        return val ** 2

# define a search space
space = hpo.hp.choice('a',
    [
        ('case 1', 1 + hpo.hp.lognormal('c1', 0, 1)),
        ('case 2', hpo.hp.uniform('c2', -10, 10))
    ])

# minimize the objective over the space
trials = hpo.Trials()
best = nu_fmin(objective, space, algo=hpo.tpe.suggest, max_evals=100, trials=trials)

print(best)
# print(best)
# # 베스트 hyperparameter 값 -> {'a': 1, 'c2': 0.01420615366247227}
# print(space_eval(space, best))
# # -> ('case 2', 0.01420615366247227}
# trials.best_trial['result']
# # -> 베스트 loss 값
# print(trials.results[0]['loss'])
# # -> 1.3010621119448424
# print(trials.vals)
# # -> hp 값들
# print(trials.results)
# # -> loss 값
# print(trials.results[0]['loss'])
# # -> 0번째 시도에서 loss 값?