from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from itertools import product
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.svm import SVC
from kdd99 import KDD99Model
from notifier import LineClient

logger.add('logs/ex_kdd99.log', rotation='5 MB', retention='10 days', level='INFO')


def main():
    logger.info(f"main: cpu_count: {cpu_count()} used: {max(1, cpu_count() - 2)}")
    layers_patterns = (
        [],
        [20, 10, 5],
        [20, 15, 10, 5],
        [20, 15, 10],
    )
    ae_used_datas = ('all', 'normal', 'u2r')
    dropped_patterns = (True, False)

    default_params = {
        'use_full': False,
        'debug': False,
        'encoder_param': {
            'epochs': 10,
            'activation': 'relu',
            'batch_size': 32,
        },
        'splits': 4,
        'random_seed': 2023,
    }

    with ProcessPoolExecutor(max_workers=max(1, cpu_count() - 3)) as executor:
        futures = dict()
        for ae_used_data, layers, dropped in product(ae_used_datas, 
                                                    layers_patterns, 
                                                    dropped_patterns):
            
            default_params['ae_used_data'] = ae_used_data
            default_params['encoder_param']['layers'] = layers
            default_params['dropped'] = dropped


            # LogisticRegression
            params = default_params.copy()
            params['model_param'] = dict(solver='lbfgs', max_iter=200)
            params['model_name'] = "LogisticRegression"
            model = KDD99Model(LogisticRegression, **params)
            future = executor.submit(model.run)
            futures[f"{params['model_name']}{layers}{dropped}{ae_used_data}"] = future

            # Support Vector Machine
            C_patterns = (0.1, 1.0, 10.0)
            for C in C_patterns:
                params = default_params.copy()
                params['model_name'] = 'SVC'
                params['model_param'] = dict(kernel='rbf', gamma='scale', C=C) # TODO:  gamma='scale', C=1.0
                model = KDD99Model(SVC, **params)
                future = executor.submit(model.run)
                futures[f"{params['model_name']}{layers}{dropped}{ae_used_data}"] = future

            # RandomForestClassifier
            params = default_params.copy()
            params['model_name'] = 'RandomForestClassifier'
            params['model_param'] = {'n_estimators': 1000, 'verbose': 0, 'warm_start': False, 'ccp_alpha': 0.0}
            model = KDD99Model(RandomForestClassifier, **params)
            model.output["importances"] = dict()
            # どの特徴に重きを置いているか調べる
            def check_importances(x_test, y_test, y_pred, _model, *_):
                for k ,v in zip(x_test.columns, _model.feature_importances_):
                    if k in model.output["importances"]:
                        model.output["importances"][k] += v
                    else:
                        model.output["importances"][k] = v
            model.additional_metrics = check_importances
            executor.submit(model.run)
            futures[f"{params['model_name']}{layers}{dropped}{ae_used_data}"] = future
        for k in futures:
            try:
                logger.info(f"{k} finished: {futures[k].result()}")
            except Exception as e:
                logger.error(f"{k}: {e}")

    LineClient().send_text("finished")


if __name__ == '__main__':
    main()
