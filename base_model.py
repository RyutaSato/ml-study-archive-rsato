
from abc import ABC, abstractmethod
from datetime import datetime as dt, timezone, timedelta
import os
from socket import MsgFlag
from typing import Optional
import warnings
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from general_utils import generate_encoder, insert_results
from notifier import LineClient
import keras
VERSION = '1.1.1'
logger.add('logs/base_model.log', rotation='5 MB', retention='10 days', level='INFO')
ROOT_DIR = os.getcwd()
warnings.simplefilter('ignore')

class ModelBase(ABC):

    def __init__(self, **config) -> None:
        self._current_task: str = 'not_started'
        self.start_time: dt = dt.now(tz=timezone(timedelta(hours=9))) # 実際には、runが呼ばれた時点での時刻
        self.config: dict = config
        self.Model = None
        self.debug: bool = False if config['debug'] is None else config['debug']
        self.random_seed: int = config['random_seed']
        self.splits: int = config['splits']
        self.ae_used_data = config['ae_used_data'] # all or specific class
        self.model_name: str = config['model_name']
        self.encoder_param: dict = config['encoder_param']
        self.layers: list[int] = self.encoder_param['layers']
        self.model_param: dict = config['model_param']
        self.model_param['random_state'] = self.random_seed
        self.y_pred: Optional[pd.Series] = None
        self.x: Optional[pd.DataFrame] = None
        self.x_preprocessed: Optional[pd.DataFrame] = None
        self.x_new_features: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.encoder: Optional[keras.Sequential] = None
        self.confusion_matrix: Optional[pd.DataFrame] = None
        self.scores = list()

        self.labels: Optional[list[str]] = None 
        self.correspondence: Optional[dict[str, int]] = None
        self.output = {
            'random_seed': self.random_seed,
            'splits': self.splits,
            'model_name': self.model_name,
            'encoder_param': self.encoder_param,
            'model_param': self.model_param, 
            'result': dict()
            }
        logger.info(f"started: {self.model_name} {self.layers} {self.ae_used_data}")


        assert type(self.config) is dict, f"config is {type(self.config)}"
        assert type(self.random_seed) is int, f"random_seed is {type(self.random_seed)}"
        assert type(self.splits) is int, f"splits is {type(self.splits)}"
        assert type(self.model_name) is str, f"model_name is {type(self.model_name)}"
        assert type(self.encoder_param) is dict, f"encoder_param is {type(self.encoder_param)}"
        assert type(self.layers) is list, f"config['layers'] is {type(self.layers)}"
        assert type(self.encoder_param['epochs']) is int
        assert type(self.encoder_param['batch_size']) is int
        assert type(self.encoder_param['activation']) is str

    @property
    def feature_size(self) -> int:
        if self.x is None:
            return 0
        if self.layers is None or self.layers == []:
            return self.x.shape[1]
        return self.x.shape[1] + self.layers[-1]
    
    @property
    def datetime(self):
        return dt.now(tz=timezone(timedelta(hours=9)))

    @property
    def elapsed_time(self) -> timedelta:
        return self.datetime - self.start_time

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def current_task(self) -> str:
        return self._current_task

    @current_task.setter
    def current_task(self, value: str) -> None:
        self._current_task = value
        logger.info(f'task started: {value}')

    @property
    def snapshot(self) -> dict:
        return {
            'name': self.name,
            'current_task': self.current_task,
            'config': self.output,
        }

    @abstractmethod
    def load(self) -> None:
        pass
    
    def preprocess(self) -> None:
        self.current_task = 'preprocess'

    def train_and_predict(self) -> None:
        self.current_task = 'train and predict'
        self.start_time: dt = dt.now(tz=timezone(timedelta(hours=9)))
        assert self.x is not None, "x is None"
        assert self.y is not None, "y is None"
        assert self.labels is not None, "labels is None"
        assert self.Model is not None, "Model is None"
        assert self.correspondence is not None, "correspondence is None"

        k_fold = StratifiedKFold(n_splits=self.splits, 
                                 shuffle=True, 
                                 random_state=self.random_seed)
        _generator = k_fold.split(self.x, self.y)
        for fold, (train_idx, test_idx) in enumerate(_generator):
            logger.info(f"phase: {fold + 1}/{self.splits}")
            # データを分割
            x_train, y_train = self.x.iloc[train_idx], self.y.iloc[train_idx]
            x_test, y_test = self.x.iloc[test_idx], self.y.iloc[test_idx]


            if self.layers:
                # エンコーダー生成
                # エンコーダーの学習に使用するデータを選択
                if self.ae_used_data == 'all':
                    x_train_ae = x_train
                else:
                    x_train_ae = x_train[y_train == self.correspondence[self.ae_used_data]]
                _encoder = generate_encoder(x_train_ae, **self.encoder_param)
                # 新たな特徴量を生成
                x_train_new_features = pd.DataFrame(
                    _encoder.predict(x_train, verbose="0"),
                    columns=[f"ae_{idx}" for idx in range(self.layers[-1])],
                    index=x_train.index
                    )
                x_test_new_features = pd.DataFrame(
                    _encoder.predict(x_test, verbose="0"),
                    columns=[f"ae_{idx}" for idx in range(self.layers[-1])],
                    index=x_test.index
                    )

                # データを結合
                x_train = pd.concat([x_train, x_train_new_features], axis=1)
                x_test = pd.concat([x_test, x_test_new_features], axis=1)

            # モデルの初期化
            _model = self.Model(**self.model_param)
            # モデルを学習
            _model.fit(x_train, y_train)

            # 予測
            y_pred = pd.Series(_model.predict(x_test), index=y_test.index)

            # テストデータで評価
            accuracy: dict = classification_report(y_test, y_pred, output_dict=True) # type: ignore
            self.additional_metrics(x_test, y_test, y_pred, _model)
            self.scores.append(accuracy)
            self.confusion_matrix += confusion_matrix(y_test, y_pred, labels=range(len(self.labels)))

    def additional_metrics(self, x_test, y_test, y_pred, _model, *_):
        pass

    def aggregate(self) -> None:
        self.current_task = 'aggregate'

        assert type(self.labels) is list, f"labels is {type(self.labels)}"
        assert type(self.confusion_matrix) is pd.DataFrame, f"confusion_matrix is {type(self.confusion_matrix)}"

        s_labels: dict[str, str] = {
            **{str(k): v for k, v in enumerate(self.labels)},
            "macro avg": "macro avg"
        }

        # 精度の集計
        for sl in s_labels:
            self.output['result'][s_labels[sl]] = dict()
            if not hasattr(self.scores[0][sl], 'keys'):
                continue
            for k2 in self.scores[0][sl].keys():
                if k2 == 'support':
                    self.output['result'][s_labels[sl]][k2] = int(np.sum([self.scores[i][sl][k2] for i in range(self.splits)]))
                else:
                    self.output['result'][s_labels[sl]][k2] = np.mean([self.scores[i][sl][k2] for i in range(self.splits)]).round(4)



        # 混同行列の集計
        confusion_dict = self.confusion_matrix.T.rename(
            columns={idx: val for idx, val in enumerate(self.labels)}, 
            index={idx: val for idx, val in enumerate(self.labels)}
        ).to_dict()
        for true_label, pred_labels in confusion_dict.items():
            for pred_label, value in pred_labels.items():
                self.output['result'][true_label]["pred_" + pred_label] = value
        
        # 特徴量の数を出力に追加
        self.output['dataset']['total_feature'] = self.feature_size
        self.output['dataset']['default_feature'] = self.x.shape[1] # type: ignore
        self.output['dataset']['ae_feature'] = self.layers[-1] if self.layers else 0

        self.output['datetime'] = self.datetime
        self.output['elapsed_time'] = str(self.elapsed_time)
        self.output['version'] = VERSION


        # 結果を出力
        if not self.debug:
            insert_results(self.output)

    
    def send_status(self, action: str, ) -> None:
        pass


    def send_error(self, error) -> None:
        line_client = LineClient()
        line_client.send_dict({
            "error": error.__str__(),
            "snapshot": self.snapshot,
        })
    

    def run(self) -> None:
        try:
            self.load()
            self.preprocess()
            self.train_and_predict()
            self.aggregate()
            logger.info(f'task finished')
        except Exception as e:
            err_msg = f"{self.model_name} {self.layers} error: {e}"
            logger.error(err_msg)
            self.send_error(err_msg)
            raise e
    

if __name__ == '__main__':
    pass
