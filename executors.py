from concurrent.futures import Future, Executor
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def lr_executor(default_params, executor: Executor, Flow) -> Future:
    params = default_params.copy()
    params['model_param'] = dict(solver='lbfgs',
                                 max_iter=100,
                                 penalty='l2',
                                 dual=False,
                                 tol=1e-4,
                                 C=1.0,
                                 fit_intercept=True,
                                 intercept_scaling=1, )
    params['model_name'] = "LogisticRegression"
    flow = Flow(LogisticRegression, **params)
    future = executor.submit(flow.run)
    return future


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
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    params['model_name'] = "LightGBM"
    flow = Flow(LGBMClassifier, **params)
    future = executor.submit(flow.run)
    return future


def lgb_optuna_executor(default_params, executor: Executor, Flow) -> Future:
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
    }
    params['model_name'] = "LightGBM+optuna"
    flow = Flow(LGBMClassifier, **params)
    future = executor.submit(flow.run)
    return future


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
    params['model_param'] = dict(C=1.0,
                                 kernel='rbf',
                                 degree=3,
                                 gamma='scale',
                                 coef0=0.0,
                                 shrinking=True,
                                 probability=False,
                                 tol=1e-3,
                                 cache_size=200,
                                 verbose=False,
                                 max_iter=-1,
                                 decision_function_shape='ovr', )
    params['model_name'] = "SVC"
    flow = Flow(SVC, **params)
    future = executor.submit(flow.run)
    return future


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
    params['model_param'] = dict(n_estimators=100,
                                 criterion='gini',
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 min_weight_fraction_leaf=0.0,
                                 bootstrap=True,
                                 oob_score=False,
                                 ccp_alpha=0.0, )
    flow = Flow(RandomForestClassifier, **params)
    future = executor.submit(flow.run)
    return future


def mp_executor(default_params, executor: Executor, Flow) -> Future:
    params = default_params.copy()
    params['model_name'] = 'MultiPerceptron'
    params['model_param'] = dict(activation='relu',
                                 hidden_layer_sizes=(15, 10, 5),
                                 max_iter=200,
                                 solver='adam',
                                 alpha=0.0001,
                                 batch_size='auto',
                                 learning_rate='constant',
                                 learning_rate_init=0.001,
                                 power_t=0.5,
                                 shuffle=True,
                                 tol=1e-4,
                                 verbose=False,
                                 warm_start=False,
                                 momentum=0.9,
                                 nesterovs_momentum=True,
                                 early_stopping=True,  # default False
                                 validation_fraction=0.1,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=1e-8,
                                 n_iter_no_change=10,
                                 max_fun=15000
                                 )
    flow = Flow(MLPClassifier, **params)
    future = executor.submit(flow.run)
    return future


