from hyperopt.base import STATUS_STRINGS
from hyperopt.base import STATUS_NEW
from hyperopt.base import STATUS_RUNNING
from hyperopt.base import STATUS_SUSPENDED
from hyperopt.base import STATUS_OK
from hyperopt.base import STATUS_FAIL

from hyperopt.base import JOB_STATES
from hyperopt.base import JOB_STATE_NEW
from hyperopt.base import JOB_STATE_RUNNING
from hyperopt.base import JOB_STATE_DONE
from hyperopt.base import JOB_STATE_ERROR

from hyperopt.base import Ctrl
from hyperopt.base import Trials
from hyperopt.base import trials_from_docs
from hyperopt.base import Domain

# -- syntactic sugar
from hyperopt import hp

# -- exceptions
from hyperopt import exceptions

# -- Import built-in optimization algorithms
from hyperopt import rand
from hyperopt import tpe
from hyperopt import atpe
from hyperopt import mix
from hyperopt import anneal

# -- spark extension
from hyperopt.spark import SparkTrials

from hyperopt.fmin import fmin #as nu_fmin
from hyperopt.fmin import fmin_pass_expr_memo_ctrl
from hyperopt.fmin import FMinIter
from hyperopt.fmin import partial
from hyperopt.fmin import space_eval

from .nu_importance import calculate_importance
import numpy as np
import json

def nu_fmin(objective, space, algo, max_evals, trials, rseed=1337, full_model_string=None, notebook_name=None, verbose=True, stack=3, keep_temp=False, data_args=None):
    best = fmin(objective, space, algo=algo, max_evals=max_evals, trials=trials, rstate=np.random.RandomState(rseed), return_argmin=True)
    importances = calculate_importance(trials)
    print("====================importance====================")
    print(importances)
    # 넣을 것 : algo, max_evals, space, trial_results, trials_vals, best_loss, best_hp, 
    # all_info_dict = dict()
    # all_info_dict["method"] = algo
    # # all_info_dict["config"] = space
    # all_info_dict["best_result"] = trials.best_trial['result'] 
    # all_info_dict["best_hp"] = best
    # all_info_dict["trial_result"] = trials.results 
    # all_info_dict["trial_hp"] = trials.vals
    # all_info = json.dumps(all_info_dict)
    # asyncio.run(Requests().post_action(request_datas = all_info_dict, url = "http://localhost:7000/admin/sdk/"))
    return best
