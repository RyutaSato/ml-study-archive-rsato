import yaml
from creditcardfraud import CreditCardFraudFlow
from imb_data import ImbalancedDatasetFlow
from kdd99 import KDD99Flow

import executors as _exe


def load_config():
    with open('main.yml', 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


runners = {
    'lr': _exe.lr_executor,
    'lgb': _exe.lgb_executor,
    'svm': _exe.svm_executor,
    'rf': _exe.rf_executor,
    'mp': _exe.mp_executor,
    'lr_optuna': _exe.lr_optuna_executor,
    'lgb_optuna': _exe.lgb_optuna_executor,
    'svm_optuna': _exe.svm_optuna_executor,
    'rf_optuna': _exe.rf_optuna_executor,
    'mp_optuna': _exe.mp_optuna_executor,
}

flows = {
    'creditcardfraud': CreditCardFraudFlow,
    'kdd99': KDD99Flow,
    'imbalance': ImbalancedDatasetFlow,
}
