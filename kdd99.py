from tabnanny import check
from unittest.mock import Base
from lightgbm import LGBMClassifier
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from base_flow import BaseFlow, ROOT_DIR

# 除外する特徴量のリスト
ignore_names = [
    "hot", "num_compromised", "num_file_creations",
    "num_outbound_cmds", "is_host_login", "srv_count",
    "srv_serror_rate", "srv_rerror_rate", "same_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_diff_srv_rate"
]
category_names = ["protocol_type", "service", "flag"]


class KDD99Flow(BaseFlow):
    """
    KDD'99データセットを用いたモデル
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
        self.use_full: bool = config['use_full']
        self.dropped = config['dropped']
        self.Model = Model
        self.labels = ['normal', 'dos', 'probe', 'r2l', 'u2r']
        self.correspondence = {label: idx for idx, label in enumerate(self.labels)}
        self.conf_matrix = pd.DataFrame(np.zeros((len(self.labels), len(self.labels)), dtype=np.int32),
                                        dtype=pd.Int32Dtype)

        self.output['dataset'] = {
            'name': self.name,
            'use_full': self.use_full,
            'dropped': self.dropped,
            'ae_used_data': self.ae_used_data,
        }

    @property
    def name(self) -> str:
        return 'kdd99'

    def load(self) -> None:
        self.current_task = 'load'

        assert type(self.correspondence) is dict, f"correspondence is {type(self.correspondence)}"

        # KDD'99 ラベルデータの読み込み
        with open(ROOT_DIR + "/datasets/kddcup.names", "r") as f:
            # 一行目は不要なので無視
            _ = f.readline()
            # `:`より手前がラベルなので，その部分を抽出してリストに追加
            names = [line.split(':')[0] for line in f]
        # 　正解ラベルを追加
        names.append("true_label")

        # KDD'99 クラスラベルデータの読み込み
        with open(ROOT_DIR + "/datasets/training_attack_types", "r") as f:
            lines = f.read().split("\n")
            classes = {'normal': self.correspondence['normal']}
            for line in lines:
                if len(line) == 0:
                    continue
                k, v = tuple(line.split(" "))
                classes[k] = self.correspondence[v]
        # KDD'99 データの読み込み
        if self.use_full:
            df = pd.read_csv(ROOT_DIR + "/datasets/kddcup.data", names=names, index_col=False)
        else:
            df = pd.read_csv(ROOT_DIR + "/datasets/kddcup.data_10_percent", names=names, index_col=False)

        # カテゴリー特徴量を削除
        data_x: pd.DataFrame = df.copy().drop(columns=category_names, axis=1)

        # 除外する特徴量を削除
        if self.dropped:
            data_x = data_x.drop(columns=ignore_names, axis=1)

        # ラベルデータを切り分ける
        data_y = data_x.pop("true_label").map(lambda x: x.replace('.', ''))

        # namesを更新
        names = data_x.columns

        # # 正規化
        # data_x = pd.DataFrame(StandardScaler().fit_transform(data_x), columns=names)

        # ラベルを変換
        data_y = data_y.map(lambda x: classes[x])

        self.x = data_x
        self.y = data_y


if __name__ == "__main__":

    params = {
        'use_full': False,
        'dropped': True,
        'debug': True,
        'ae_used_data': 'u2r',
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
        'model_name': 'RandomForest',  # 'RandomForestClassifier', # LogisticRegression, LightGBM
        'random_seed': 2023,
    }
    flow = KDD99Flow(RandomForestClassifier, **params)
    flow.run()
        
    # model.run()
    with open(ROOT_DIR + "/logs/kdd99.json", "w") as f:
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
