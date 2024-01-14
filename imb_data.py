from multiprocessing import Lock

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from base_flow import BaseFlow, ROOT_DIR
from imblearn.datasets import fetch_datasets

from schemas import Params


class ImbalancedDatasetFlow(BaseFlow):
    """
    scikit-learnの不均衡データセットを用いたモデル
    Args:
        model: モデルのクラス

    """

    def __init__(self, model, gpu_lock: Lock, _params: Params) -> None:
        super().__init__(model, gpu_lock, _params)
        self.Model = model
        self.labels = ['majority', 'minority']
        self.correspondence = {label: idx for idx, label in enumerate(self.labels)}
        self.conf_matrix = pd.DataFrame(np.zeros((len(self.labels), len(self.labels)), dtype=int), dtype=int)

    def load(self) -> None:
        self.current_task = 'load'

        assert type(self.correspondence) is dict, f"correspondence is {type(self.correspondence)}"

        # データの読み込み
        datasets = fetch_datasets(data_home=ROOT_DIR + '/datasets', download_if_missing=True)
        
        # 変換用ラベル
        classes = {-1: 0, 1: 1}
        self.x = pd.DataFrame(datasets[self.dataset.name]['data'])

        self.x.columns = self.x.columns.astype(str)
        self.y = pd.Series(datasets[self.dataset.name]['target']).map(lambda x: classes[x])


if __name__ == "__main__":
    dataset_name = 'ecoli'
    params = {
        'dataset_name': dataset_name,
        'debug': True,
        'ae_used_data': 'all',
        'encoder_param': {
            'layers': [20, 15, 10],
            'epochs': 1,
            'activation': 'relu',
            'batch_size': 32,
        },
        'model_param': {
            # RandomForest
            'n_estimators': 10,
            'verbose': 0,
            'warm_start': False,
            # 'objective':'multiclass',
            # 'metric':'multi_logloss',
            # 'n_estimators':1000,
            # 'verbosity': -1,
        },
        'splits': 4,
        'model_name': 'RandomForest',  # 'RandomForestClassifier', # LogisticRegression, LGBM
        'random_seed': 2023,
    }
    flow = ImbalancedDatasetFlow(RandomForestClassifier, **params)
    flow.run()
        
    # model.run()
    with open(ROOT_DIR + f"/logs/{dataset_name}.json", "w") as f:
        import json


        def default(o):
            if hasattr(o, "isoformat"):
                return o.isoformat()
            elif type(o) is np.int32:
                return int(o)
            elif type(o) is np.float16:
                return float(o)
            else:
                return str(o)
        json.dump(flow.output, f, indent=4, default=default)
