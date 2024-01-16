from multiprocessing import Queue
from multiprocessing.synchronize import Lock

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from base_flow import BaseFlow, ROOT_DIR
from schemas import Params

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

    """

    def __init__(self, model, gpu_lock: Lock, _params: Params) -> None:
        super().__init__(model, gpu_lock, _params)
        if _params.dataset.name == 'kdd99_dropped':
            self.dropped = True
        else:
            self.dropped = False
        self.Model = model
        # self.labels = ['normal', 'dos', 'probe', 'r2l', 'u2r']
        # alias
        self.labels = ['majority', 'dos', 'probe', 'r2l', 'minority']
        self.correspondence = {label: idx for idx, label in enumerate(self.labels)}
        self.conf_matrix = pd.DataFrame(np.zeros((len(self.labels), len(self.labels)), dtype=np.int32),
                                        dtype=pd.Int32Dtype)

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

        # 各カテゴリに属する攻撃タイプのリスト
        dos_attacks = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop']
        probe_attacks = ['ipsweep', 'nmap', 'portsweep', 'satan']
        r2l_attacks = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster']
        u2r_attacks = ['buffer_overflow', 'loadmodule', 'perl', 'rootkit']
        # 辞書の初期化
        kdd99_mapping = {'normal': 0}

        # 各カテゴリの攻撃タイプに対応するラベルを辞書に追加
        for attack in dos_attacks:
            kdd99_mapping[attack] = 1
        for attack in probe_attacks:
            kdd99_mapping[attack] = 2
        for attack in r2l_attacks:
            kdd99_mapping[attack] = 3
        for attack in u2r_attacks:
            kdd99_mapping[attack] = 4

        # KDD'99 データの読み込み
        df = pd.read_csv(ROOT_DIR + "/datasets/kddcup.data_10_percent", names=names, index_col=False)

        # カテゴリー特徴量を削除
        data_x: pd.DataFrame = df.copy().drop(columns=category_names, axis=1)

        # 除外する特徴量を削除
        if self.dropped:
            data_x = data_x.drop(columns=ignore_names, axis=1)

        # ラベルデータを切り分ける
        data_y = data_x.pop("true_label").map(lambda x: x.replace('.', ''))

        # ラベルを変換
        data_y = data_y.map(lambda x: kdd99_mapping[x])

        self.x = data_x
        self.y = data_y


if __name__ == "__main__":

    params = Params(
        hash="",
        dataset={
            "name": "kdd99_dropped",
            "standardization": True,
        },
        model={
            "name": "RandomForest",
            "params": {},
            "optuna": False,
        },
        ae={
            "layers": [20, 10, 5],
            "used_class": "all",
        },
        env={
            "version": "2.0.0",
            "datetime": "2021-05-08T15:00:00",
            "elapsed_time": "00:00:00",
        },
        result={},
    )
    flow = KDD99Flow(RandomForestClassifier, params)
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
