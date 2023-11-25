from unittest.mock import Base
from lightgbm import LGBMClassifier
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from base_flow import BaseFlow, ROOT_DIR


class CreditCardFraudFlow(BaseFlow):
    """
    Credit Card Fraudデータセットを用いたモデル
    Args:
        model: モデルのクラス
        use_full: データセットの全データを使うかどうか
        dropped: 除外する特徴量のリスト
    Keys:
        - use_full: データセットの全データを使うかどうか
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
        self.labels = ['normal', 'anomaly']
        self.correspondence = {label: idx for idx, label in enumerate(self.labels)}
        self.conf_matrix = pd.DataFrame(np.zeros((len(self.labels), len(self.labels)), dtype=np.int32),
                                        dtype=pd.Int32Dtype)

        self.output['dataset'] = {
            'name': self.name,
            'ae_used_data': self.ae_used_data,
        }

    @property
    def name(self) -> str:
        return 'creditcardfraud'

    def load(self) -> None:
        self.current_task = 'load'

        assert type(self.correspondence) is dict, f"correspondence is {type(self.correspondence)}"


        # データの読み込み
        data_x = pd.read_csv(ROOT_DIR + "/datasets/creditcard.csv")


        # ラベルデータを切り分ける
        data_y = data_x.pop("Class")
        
        # namesを更新
        names = data_x.columns

        # 正規化
        data_x = pd.DataFrame(StandardScaler().fit_transform(data_x), columns=names)

        # ラベルを変換
        data_y = data_y.map(lambda x: int(x))   

        self.x = data_x
        self.y = data_y
    
    
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    params = {
        # 'model': LogisticRegression,
        'debug': True,
        'encoder_param': {
            'layers': [10],
            'epochs': 2,
            'batch_size': 32,
            'activation': 'relu',
        },
        'model_param': {
            'max_iter': 10,
        },
        'splits': 5,
        'model_name': 'LightGBM',
        'random_seed': 2023,
        'ae_used_data': 'all',
    }
    flow = CreditCardFraudFlow(LGBMClassifier,**params)
    flow.run()
    with open(ROOT_DIR + "/logs/creditcardfraud.json", "w") as f:
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

