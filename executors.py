
from concurrent.futures import Future, Executor
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



def lr_executor(default_params, executor: Executor, Flow) -> Future:
    params = default_params.copy()
    params['model_param'] = dict(solver='lbfgs', max_iter=200)
    params['model_name'] = "LogisticRegression"
    flow = Flow(LogisticRegression, **params)
    return executor.submit(flow.run)

def lgb_executor(default_params, executor: Executor, Flow) -> Future:
    """
    与えられたパラメータを用いてLightGBMを実行します。

    Args:
        default_params (dict): モデルのデフォルトパラメータ。
        executor (Executor): 並行処理のためのExecutor。
        Flow (Flow): モデル実行のためのFlowクラス。

    Returns:
        Future: 実行のFutureオブジェクト。
    """
    params = default_params.copy()
    params['model_param'] = {
        # 'objective': 'multiclass',
        # 'num_class': 3,  # クラスの数
        # 'metric': 'multi_logloss',
        # 'boosting_type': 'gbdt',
        # 'num_leaves': 31,
        # 'learning_rate': 0.05,
        # 'feature_fraction': 0.9
    }
    params['model_name'] = "LightGBM"
    flow = Flow(LGBMClassifier, **params)
    return executor.submit(flow.run)

def svm_executor(default_params, executor: Executor, Flow) -> Future:
    """
    与えられたパラメータを用いてSVMを実行します。

    Args:
        default_params (dict): モデルのデフォルトパラメータ。
        executor (Executor): 並行処理のためのExecutor。
        Flow (Flow): モデル実行のためのFlowクラス。

    Returns:
        Future: 実行のFutureオブジェクト。
    """
    params = default_params.copy()
    params['model_param'] = dict()
    params['model_name'] = "SVC"
    flow = Flow(SVC, **params)
    return executor.submit(flow.run)

def rf_executor(default_params, executor: Executor, Flow) -> Future:
    """
    与えられたパラメータを用いてランダムフォレストを実行します。

    Args:
        default_params (dict): モデルのデフォルトパラメータ。
        executor (Executor): 並行処理のためのExecutor。
        Flow (Flow): モデル実行のためのFlowクラス。

    Returns:
        Future: 実行のFutureオブジェクト。
    """
    params = default_params.copy()
    params['model_name'] = 'RandomForest'
    params['model_param'] = {'n_estimators': 1000, 'verbose': 0, 'warm_start': False, 'ccp_alpha': 0.0}
    flow = Flow(RandomForestClassifier, **params)
    return executor.submit(flow.run)

def mp_executor(default_params, executor: Executor, Flow) -> Future:
    params = default_params.copy()
    params['model_name'] = 'MultiPerceptron'
    params['model_param'] = dict(activation='relu', hidden_layer_sizes=(20, 15, 10), max_iter=500)
    flow = Flow(MLPClassifier, **params)
    return executor.submit(flow.run)
