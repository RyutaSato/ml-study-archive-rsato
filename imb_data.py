from lightgbm import LGBMClassifier
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from base_flow import BaseFlow, ROOT_DIR
from imblearn.datasets import fetch_datasets


class ImbalancedDatasetFlow(BaseFlow):
    """
    scikit-learnの不均衡データセットを用いたモデル
    Args:
        model: モデルのクラス
        use_full: データセットの全データを使うかどうか
        dropped: 除外する特徴量のリスト
    Keys:
        - use_full: データセットの全データを使うかどうか
        - dropped: 除外する特徴量のリスト
        - autoencoder: オートエンコーダーのパラメータ
        - model: モデルのパラメータ
        - splits: データセットの分割数
        - dataset: データセットの情報
        - model_name: モデルの名前
        - random_seed: ランダムシード
        - ae_used_data: オートエンコーダーに使うデータの種類



    """

    def __init__(
            self,
            Model,
            **config) -> None:
        super().__init__(**config)
        self.Model = Model
        self.labels = ['majority', 'minority']
        self.correspondence = {label: idx for idx, label in enumerate(self.labels)}
        self.conf_matrix = pd.DataFrame(np.zeros((len(self.labels), len(self.labels)), dtype=int), dtype=int)
        self.dataset_name = config['dataset_name']
        self.output['dataset'] = {
            'name': self.name,
            'ae_used_data': self.ae_used_data,
        }

    @property
    def name(self) -> str:
        return self.dataset_name

    def load(self) -> None:
        self.current_task = 'load'

        assert type(self.correspondence) is dict, f"correspondence is {type(self.correspondence)}"

        # データの読み込み
        datasets = fetch_datasets(data_home=ROOT_DIR + '/datasets', download_if_missing=True)
        
        # 変換用ラベル
        classes = {-1: 0, 1: 1}
        self.x = pd.DataFrame(datasets[self.name]['data'])

        self.x.columns = self.x.columns.astype(str)
        self.y = pd.Series(datasets[self.name]['target']).map(lambda x: classes[x])


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
