from asyncio import futures
from concurrent.futures import ProcessPoolExecutor, as_completed
from http import client
from multiprocessing import cpu_count
from itertools import product
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from creditcardfraud import CreditCardFraudModel
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
    ae_used_datas = ('all', 'normal', 'anomaly')
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

    with ProcessPoolExecutor(max_workers=max(1, cpu_count() - 2)) as executor:
        futures = dict()
        for as_used_data, layers, dropped in product(ae_used_datas, 
                                                    layers_patterns, 
                                                    dropped_patterns):
            # LogisticRegression
            params = default_params.copy()
            params['ae_used_data'] = as_used_data
            params['encoder_param']['layers'] = layers
            params['dropped'] = dropped
            params['model_param'] = {
                'solver': 'lbfgs',
                'max_iter': 200,
            }
            params['model_name'] = 'LogisticRegression'
            model = CreditCardFraudModel(LogisticRegression, **params)
            future = executor.submit(model.run)
            futures[f"{params['model_name']}{layers}{dropped}{as_used_data}"] = future


            # Support Vector Machine
            params = default_params.copy()
            params['ae_used_data'] = as_used_data
            params['encoder_param']['layers'] = layers
            params['dropped'] = dropped
            params['model_name'] = 'SVC'
            params['model_param'] = {
                'kernel': 'rbf',
                'gamma': 'scale',
                'C': 1.0,

            }
            model = CreditCardFraudModel(SVC, **params)
            future = executor.submit(model.run)
            futures[f"{params['model_name']}{layers}{dropped}{as_used_data}"] = future

            # RandomForestClassifier
            params = default_params.copy()
            params['ae_used_data'] = as_used_data
            params['encoder_param']['layers'] = layers
            params['dropped'] = dropped
            params['model_name'] = 'RandomForestClassifier'
            params['model_param'] = {
                'n_estimators': 100,
                'verbose': 0,
                'warm_start': False,
                'ccp_alpha': 0.0,
            }
            model = CreditCardFraudModel(RandomForestClassifier, **params)
            executor.submit(model.run)
            futures[f"{params['model_name']}{layers}{dropped}{as_used_data}"] = future
        for k in futures:
            try:
                logger.info(f"finished: {futures[k].result()}")
            except Exception as e:
                logger.error(e)
                
    LineClient().send_text("finished")

if __name__ == '__main__':
    main()