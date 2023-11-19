
import yaml
from creditcardfraud import CreditCardFraudFlow
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
}

flows = {
    'creditcardfraud': CreditCardFraudFlow,
    'kdd99': KDD99Flow,
}

