import json

import yaml
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from creditcardfraud import CreditCardFraudFlow
from imb_data import ImbalancedDatasetFlow
from kdd99 import KDD99Flow
from multiprocessing import Queue, Lock

import executors as _exe
from schemas import Params
from loguru import logger


def load_config():
    with open('config.yml', 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def worker(_que: Queue, _lock: Lock):
    while True:
        params: Params = _que.get()
        logger.info(f"current queue waiting: {_que.qsize()}")
        if params is None:
            break
        _Flow = flows.get(params.dataset.name)
        _Model = models.get(params.model.name)
        if _Flow is None or _Model is None:
            logger.error(f"Flow or Model is None. Flow: {_Flow}, Model: {_Model}")
        flow = _Flow(_Model, _lock, params)
        try:
            flow.run()
        except Exception as e:
            with open("results/not_finished.json", "a") as f:
                json.dump(params.json(), f, indent=4)
        del flow, _Flow, _Model, params


# DEPRECATED
# runners = {
#     'lr': _exe.lr_executor,
#     'lgb': _exe.lgb_executor,
#     'svm': _exe.svm_executor,
#     'rf': _exe.rf_executor,
#     'mp': _exe.mp_executor,
#     'lr_optuna': _exe.lr_optuna_executor,
#     'lgb_optuna': _exe.lgb_optuna_executor,
#     'svm_optuna': _exe.svm_optuna_executor,
#     'rf_optuna': _exe.rf_optuna_executor,
#     'mp_optuna': _exe.mp_optuna_executor,
# }

flows = {
    'creditcardfraud': CreditCardFraudFlow,
    'kdd99': KDD99Flow,
    'kdd99_dropped': KDD99Flow,
    'ecoli': ImbalancedDatasetFlow,
    'optical_digits': ImbalancedDatasetFlow,
    'satimage': ImbalancedDatasetFlow,
    'pen_digits': ImbalancedDatasetFlow,
    'abalone': ImbalancedDatasetFlow,
    'sick_euthyroid': ImbalancedDatasetFlow,
    'spectrometer': ImbalancedDatasetFlow,
    'car_eval_34': ImbalancedDatasetFlow,
    'isolet': ImbalancedDatasetFlow,
    'us_crime': ImbalancedDatasetFlow,
    'yeast_ml8': ImbalancedDatasetFlow,
    'scene': ImbalancedDatasetFlow,
    'libras_move': ImbalancedDatasetFlow,
    'thyroid_sick': ImbalancedDatasetFlow,
    'coil_2000': ImbalancedDatasetFlow,
    'arrhythmia': ImbalancedDatasetFlow,
    'solar_flare_m0': ImbalancedDatasetFlow,
    'oil': ImbalancedDatasetFlow,
    'car_eval_4': ImbalancedDatasetFlow,
    'wine_quality': ImbalancedDatasetFlow,
    'letter_img': ImbalancedDatasetFlow,
    'yeast_me2': ImbalancedDatasetFlow,
    'webpage': ImbalancedDatasetFlow,
    'ozone_level': ImbalancedDatasetFlow,
    'mammography': ImbalancedDatasetFlow,
    'protein_homo': ImbalancedDatasetFlow,
    'abalone_19': ImbalancedDatasetFlow,
}

models = {
    "lr": LogisticRegression,
    "lgb": LGBMClassifier,
    "svm": SVC,
    "rf": RandomForestClassifier,
    "mp": MLPClassifier,
}
