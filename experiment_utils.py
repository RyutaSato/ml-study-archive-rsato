

from concurrent.futures import Future, Executor
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



def logistic_regression_executor(default_params, executor: Executor, Flow) -> Future:
    params = default_params.copy()
    params['model_param'] = dict(solver='lbfgs', max_iter=200)
    params['model_name'] = "LogisticRegression"
    flow = Flow(LogisticRegression, **params)
    return executor.submit(flow.run)

def lightgbm_executor(default_params, executor: Executor, Flow) -> Future:
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
    params['model_param'] = dict()
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

def random_forest_executor(default_params, executor: Executor, Flow) -> Future:
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
    flow.output["importances"] = dict()
    # どの特徴に重きを置いているか調べる
    def check_importances(x_test, __, ___, _model, *_):
        for k ,v in zip(x_test.columns, _model.feature_importances_):
            if k in flow.output["importances"]:
                flow.output["importances"][k] += v
            else:
                flow.output["importances"][k] = v
    flow.additional_metrics = check_importances
    return executor.submit(flow.run)

def multi_perceptron_executor(default_params, executor: Executor, Flow) -> Future:
    params = default_params.copy()
    params['model_name'] = 'NeuralNetwork'
    params['model_param'] = dict(activation='relu', hidden_layer_sizes=(20, 15, 10), max_iter=500)
    flow = Flow(MLPClassifier, **params)
    return executor.submit(flow.run)
