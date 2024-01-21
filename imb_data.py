from multiprocessing import Lock

import numpy as np
import pandas as pd
from base_flow import BaseFlow, ROOT_DIR
from imblearn.datasets import fetch_datasets

from schemas import Params


class ImbalancedDatasetFlow(BaseFlow):

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
    from datetime import datetime
    from schemas import *
    dataset_name = 'ecoli'
    params = Params(hash="", env=Environment(version='2.0.0', datetime=datetime.now()), dataset=Dataset(name=dataset_name), model=MLModel(name='rf'), ae=AEModel(), result=Result())
