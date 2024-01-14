from multiprocessing import Lock
import numpy as np
import pandas as pd
from base_flow import BaseFlow, ROOT_DIR
from schemas import Params


class CreditCardFraudFlow(BaseFlow):
    """
    Credit Card Fraudデータセットを用いたモデル
    Args:
        model: モデルのクラス

    """

    def __init__(self, model, gpu_lock: Lock, _params: Params) -> None:
        super().__init__(model, gpu_lock, _params)
        self.Model = model
        self.labels = ['majority', 'minority']
        self.correspondence = {label: idx for idx, label in enumerate(self.labels)}
        self.conf_matrix = pd.DataFrame(np.zeros((len(self.labels), len(self.labels)), dtype=np.int32),
                                        dtype=pd.Int32Dtype)

    def load(self) -> None:
        self.current_task = 'load'

        assert type(self.correspondence) is dict, f"correspondence is {type(self.correspondence)}"

        # データの読み込み
        data_x = pd.read_csv(ROOT_DIR + "/datasets/creditcard.csv")

        # ラベルデータを切り分ける
        data_y = data_x.pop("Class")
        # ラベルを変換
        data_y = data_y.map(lambda x: int(x))

        self.x = data_x
        self.y = data_y


if __name__ == "__main__":
    pass
