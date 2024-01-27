import base64
import json
from abc import ABC, abstractmethod
from datetime import datetime as dt, timedelta
import os
import traceback
from multiprocessing import Lock
from typing import Optional
from dotenv import load_dotenv

from schemas import Params, MLModel, Accuracy

import warnings
import tensorflow as tf
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
load_dotenv()
# from lightgbm import log_evaluation TODO
from lightgbm import log_evaluation, early_stopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from general_utils import fetch_h5_model, generate_encoder, insert_h5_model, insert_results
from notifier import LineClient
from tensorflow import keras
import optuna

VERSION = '2.0.0'

ROOT_DIR = os.getcwd()
warnings.simplefilter('ignore')
logger.info(f"GPU {tf.config.list_physical_devices('GPU')}")
if not tf.config.list_physical_devices('GPU'):
    logger.warning(f"GPU is not available")

DEBUG: bool = True if os.getenv('DEBUG') == 'true' else False
logger.info(f'DEBUG: {DEBUG}')


class BaseFlow(ABC):
    """
    BaseFlowは、機械学習の基本的なフローを抽象化したクラスです。

    Attributes:
        _current_task (str): 現在のタスクの状態を表す文字列。
        start_time (datetime): タスクの開始時間。
        debug (bool): デバッグモードが有効かどうかを示すフラグ。
        random_seed (int): 乱数のシード値。
        splits (int): データの分割数。
        model_param (dict): モデルのパラメータ。
        y_pred (pd.Series): 予測値。
        x (pd.DataFrame): 入力データ。
        x_preprocessed (pd.DataFrame): 前処理済みの入力データ。
        x_new_features (pd.DataFrame): 新たに生成された特徴量。
        y (pd.Series): 目的変数。
        encoder (keras.Sequential): エンコーダ。
        conf_matrix (pd.DataFrame): 混同行列。
        scores (list): スコアのリスト。
    """

    def __init__(self, model, gpu_lock: Lock, params: Params) -> None:
        if params.env.version != VERSION:
            raise ValueError(f"version is {params.env.version}. It is not supported.")
        self._current_task: str = 'not_started'
        self.start_time: dt = dt.now()
        self.env = params.env
        self.dataset = params.dataset
        self.model = params.model
        self.ae = params.ae
        self.result = params.result
        self._hash = params.hash
        self.Model = model
        self.debug: bool = DEBUG
        self.random_seed: int = 2023
        self.model.params['random_state'] = self.random_seed
        self.splits: int = 4
        self.model_param: MLModel = params.model
        self.y_pred: Optional[pd.Series] = None
        self.x: Optional[pd.DataFrame] = None
        self.x_preprocessed: Optional[pd.DataFrame] = None
        self.x_new_features: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.encoder: Optional[keras.Sequential] = None
        self.conf_matrix: Optional[pd.DataFrame] = None
        self.scores = list()
        self.gpu_lock = gpu_lock

        self.labels = None
        self.correspondence = None

    @property
    def total_feature_num(self) -> int:
        return self.x.shape[1] + self.ae.layers[-1]

    @property
    def elapsed_time(self) -> timedelta:
        return dt.now() - self.start_time

    @property
    def current_task(self) -> str:
        return self._current_task

    @current_task.setter
    def current_task(self, value: str) -> None:
        self._current_task = value
        logger.info(f'{self.dataset.name}: {self.model.name}{self.ae.layers} {self.ae.used_class} started: {value}')

    @property
    def snapshot(self) -> dict:
        return {
            "dataset": self.dataset.dict(),
            "ae": self.ae.dict(),
            "env": self.env.dict(),
            "model": self.model.dict()
        }

    def __hash__(self):
        return self._hash

    @abstractmethod
    def load(self) -> None:
        pass

    def preprocess(self) -> None:
        self.current_task = 'preprocess'
        assert self.x is not None, "x is None"
        #
        # self.x = pd.DataFrame(MinMaxScaler().fit_transform(self.x), columns=self.x.columns, index=self.x.index)
        # self.x = pd.DataFrame(StandardScaler().fit_transform(self.x), columns=self.x.columns, index=self.x.index)

    def _k_fold_preprocess(self, x_train, y_train, x_test, y_test) -> tuple[pd.DataFrame, pd.DataFrame]:
        # 標準化
        if self.dataset.standardization:
            scaler = StandardScaler()
        # 正規化
        elif self.dataset.normalization:
            scaler = MinMaxScaler()
        else:
            return x_train, x_test
        x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
        x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
        
        return x_train, x_test

    def _k_fold_generate_ae(self, x_train, y_train, x_test, y_test, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        新たな特徴量を生成します。

        このメソッドは、新たな特徴量を生成します。
        新たな特徴量の生成は、以下の手順で行われます。
        1. エンコーダーの生成
        2. 新たな特徴量の生成

        Args:
            x_train (pd.DataFrame): 訓練データの入力データ。
            y_train (pd.Series): 訓練データの目的変数。
            x_test (pd.DataFrame): テストデータの入力データ。
            y_test (pd.Series): テストデータの目的変数。

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: 新たな特徴量。

        """
        # エンコーダー生成
        # エンコーダーの学習に使用するデータを選択
        if self.ae.used_class == 'all':
            x_train_ae = x_train
        else:
            assert type(self.correspondence) is dict
            x_train_ae = x_train[y_train == self.correspondence[self.ae.used_class]]

        # モデルのファイル名
        k = self.dataset.name
        k += str(self.dataset.standardization)
        k += str(self.dataset.normalization)
        k += str(self.ae.used_class)
        k += str(self.ae.layers)
        k += str(fold)
        k += str(self.env.version)
        k: str = base64.b64encode(k.encode()).decode()
        file_name = ROOT_DIR + "/models/" + k + ".h5"
        # 保存済みモデルがある場合
        if not os.path.exists(file_name):
            fetch_h5_model(k)
        if os.path.exists(file_name):
            # 保存しているモデルの読み込み
            _encoder = keras.models.load_model(file_name, compile=False)
            _encoder.compile(optimizer='adam', loss='mse')
        # 保存済みモデルがない場合
        else:
            # エンコーダーの生成
            with self.gpu_lock:
                self.current_task = f"train phase: encoder generating"
                _encoder = generate_encoder(x_train_ae, **self.ae.dict())
            # モデルの保存
            _encoder.save(file_name)
            insert_h5_model(k)
        # 新たな特徴量を生成
        x_train_ae = pd.DataFrame(
            _encoder.predict(x_train, verbose=0),  # type: ignore
            columns=[f"ae_{idx}" for idx in range(self.ae.layers[-1])],
            index=x_train.index
        )
        x_test_ae = pd.DataFrame(
            _encoder.predict(x_test, verbose=0),  # type: ignore
            columns=[f"ae_{idx}" for idx in range(self.ae.layers[-1])],
            index=x_test.index
        )

        if self.ae.standardization:
            scaler = StandardScaler()
        
        elif self.ae.normalization:
            scaler = MinMaxScaler()
        else:
            return x_train_ae, x_test_ae
        x_train_ae = pd.DataFrame(
            scaler.fit_transform(x_train_ae), columns=x_train_ae.columns, index=x_train_ae.index)
        x_test_ae = pd.DataFrame(
            scaler.transform(x_test_ae), columns=x_test_ae.columns, index=x_test_ae.index) 
        return x_train_ae, x_test_ae

    def k_fold_cross_validation(self, only_generate_encoder=False) -> None:
        """
        モデルの訓練と予測を行います。

        このメソッドは、データセットを訓練セットとテストセットに分割し、
        モデルを訓練した後、テストセットで予測を行います。
        StratifiedKFoldを使用してデータを分割し、各分割でモデルを訓練します。

        このメソッドを実行する前に、以下のプロセスが必要です:
        1. データセットの準備: `self.x`, `self.y`, `self.labels`が設定されていること。
        2. モデルの準備: `self.Model`が設定されていること。
        3. 対応関係の準備: `self.correspondence`が設定されていること。

        Raises:
            AssertionError: 必要なデータがNoneの場合に発生します。

        """
        self.current_task = 'k fold cross validaton'
        self.start_time: dt = dt.now()
        assert self.x is not None, "x is None"
        assert self.y is not None, "y is None"
        assert self.labels is not None, "labels is None"
        assert self.Model is not None, "Model is None"
        assert self.correspondence is not None, "correspondence is None"

        k_fold = StratifiedKFold(n_splits=self.splits, shuffle=True, random_state=self.random_seed)
        _generator = k_fold.split(self.x, self.y)
        for fold, (train_idx, test_idx) in enumerate(_generator):
            self.current_task = f"train phase: {fold + 1}/{self.splits}"

            # データを分割
            x_train, y_train = self.x.iloc[train_idx], self.y.iloc[train_idx]
            x_test, y_test = self.x.iloc[test_idx], self.y.iloc[test_idx]

            # データの前処理
            x_train, x_test = self._k_fold_preprocess(x_train, y_train, x_test, y_test)

            if self.ae.layers[-1] > 0:
                # 新たな特徴量の生成
                x_train_ae, x_test_ae = self._k_fold_generate_ae(
                    x_train, y_train, x_test, y_test, fold)

                # データを結合
                x_train = pd.concat([x_train, x_train_ae], axis=1)
                x_test = pd.concat([x_test, x_test_ae], axis=1)

            # ONLY GENERATE ENCODER
            if only_generate_encoder:
                return

            # ハイパーパラメータのチューニング
            if self.model.optuna:
                self.model.params = self.optuna(x_train, y_train)
                self.model.best_params_list.append(self.model.params)

            # モデルの初期化
            assert type(self.Model) is not None, "Model is None"
            _model = self.Model(**self.model.params)
            self.model.params = _model.get_params()

            # モデルを学習
            _model.fit(x_train, y_train)

            # 予測
            y_pred = pd.Series(_model.predict(x_test), index=y_test.index)

            # テストデータで評価
            accuracy: dict = classification_report(y_test, y_pred, output_dict=True)  # type: ignore

            if hasattr(_model, "feature_importances_"):
                for k, v in zip(x_test.columns, _model.feature_importances_):
                    if k in self.result.importances:
                        self.result.importances[k] += int(v)
                    else:
                        self.result.importances[k] = int(v)

            self.scores.append(accuracy)
            self.conf_matrix += confusion_matrix(y_test, y_pred, labels=range(len(self.labels)))

    def aggregate(self) -> Params:
        """
        予測結果を集約します。

        このメソッドは、各分割での予測結果を集約し、最終的な予測結果を生成します。
        予測結果は `self.predictions` に格納されます。

        このメソッドを実行する前に、以下のプロセスが必要です:
        1. モデルの訓練と予測: `train_and_predict` メソッドが実行されていること。

        Raises:
            AssertionError: 必要なデータがNoneの場合に発生します。

        """
        self.current_task = 'aggregate'

        assert type(self.labels) is list, f"labels is {type(self.labels)}"
        assert type(self.conf_matrix) is pd.DataFrame, f"confusion_matrix is {type(self.conf_matrix)}"

        # 精度の集計
        """
        self.scores = [
                          {
                              {"0": {'precision': 0, 'recall'....},
                              {"1": {'precision': 0, 'recall'....},
                          } for i in range(4)
                  ]
        
        """

        def _aggregate_accuracy(cls) -> Accuracy:
            return Accuracy(
                precision=np.mean([self.scores[i][cls]['precision'] for i in range(self.splits)]).round(4),
                recall=np.mean([self.scores[i][cls]['recall'] for i in range(self.splits)]).round(4),
                f1=np.mean([self.scores[i][cls]['f1-score'] for i in range(self.splits)]).round(4),
                support=int(np.sum([self.scores[i][cls]['support'] for i in range(self.splits)]))
            )

        self.result.majority = _aggregate_accuracy(str(self.correspondence['majority']))
        self.result.minority = _aggregate_accuracy(str(self.correspondence['minority']))
        self.result.macro = _aggregate_accuracy('macro avg')

        # 特徴量の数を出力に追加
        self.dataset.total_feature_num = self.total_feature_num
        self.dataset.default_feature_num = self.x.shape[1]
        self.dataset.ae_feature_num = self.ae.layers[-1]
        self.dataset.sample_num = self.x.shape[0]

        self.env.datetime = dt.now()
        self.env.elapsed_time = self.elapsed_time

        _r = Params(
            hash=self.__hash__(),
            dataset=self.dataset,
            model=self.model,
            ae=self.ae,
            env=self.env,
            result=self.result
        )
        # 結果を出力
        if not self.debug:
            insert_results(json.loads(_r.json()))
        return _r

    def send_status(self, action: str, ) -> None:
        pass

    def send_error(self, error) -> None:
        if not self.debug:
            line_client = LineClient()
            line_client.send_dict({
                "error": error.__str__(),
                "snapshot": self.snapshot,
                "traceback": traceback.format_exc()
            })

    def _get_default_params(self, trial) -> dict:
        if self.model.name == 'lr':
            return dict(
                C=trial.suggest_loguniform('C', 0.001, 1000),
                solver=trial.suggest_categorical('solver', ['lbfgs', 'sag']),
                multi_class=trial.suggest_categorical('multi_class', ['ovr', 'multinomial']),
            )
        elif self.model.name == 'svm':
            return dict(
                C=trial.suggest_loguniform('C', 1, 10000),
                kernel=trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
                gamma=trial.suggest_categorical('gamma', ['scale', 'auto']),
            )
        elif self.model.name == 'rf':
            return dict(
                n_estimators=trial.suggest_int('n_estimators', 10, 1000),
                max_depth=trial.suggest_int('max_depth', 1, 100),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 100),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 100),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                bootstrap=True,
            )
        elif self.model.name == 'mp':
            return dict(
                hidden_layer_sizes=trial.suggest_categorical('hidden_layer_sizes', [(15, 10, 5), (10, 5), (5,)]),
                activation=trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu']),
                solver=trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam']),
                alpha=trial.suggest_loguniform('alpha', 0.001, 10),
                learning_rate=trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
            )
        elif self.model.name == 'lgb':
            assert type(self.y) is pd.Series, f"y is {type(self.y)}"
            num_class = self.y.nunique()
            if num_class == 2:
                num_class = 1
                objective_ = 'binary'
                metrics = 'binary_logloss'
            else:
                objective_ = 'multiclass'
                metrics = 'multi_logloss'

            return dict(
                objective=objective_,
                num_class=num_class,
                metric=metrics,
                boosting_type='gbdt',
                num_leaves=trial.suggest_int('num_leaves', 2, 100),
                learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 0.1),
                feature_fraction=trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                bagging_fraction=trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                bagging_freq=trial.suggest_int('bagging_freq', 1, 7),
                max_depth=trial.suggest_int('max_depth', 3, 20),
                lambda_l1=trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                lambda_l2=trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                # min_child_samples=trial.suggest_int('min_child_samples', 5, 100),
                random_state=self.random_seed,
                verbose=-1,
            )
        else:
            raise ValueError(f"model_name is {self.model.name}. It is not supported.")

    def optuna(self, x_train, y_train) -> dict:

        # 学習とテストに分割．学習のサンプル数は，最大30,000件に制限
        if x_train.shape[0] * (1 - 0.25) > 30_000: # ex: 100,000 * 0.75 > 30,000
            test_size = 1 - 30000 / x_train.shape[0] # ex: 0.7
            # ex: 30,000: 70,000
            logger.info(f"sample size: {x_train.shape[0]} limitted to 30,000")  
        else:
            test_size = 0.25  # default
        
        x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=test_size,random_state=self.random_seed,stratify=y_train)

        def objective(trial):
            self.current_task = f'optuna phase {trial.number + 1} / 100'
            # ハイパーパラメータの設定
            best_params = self._get_default_params(trial)

            # モデルの学習
            assert self.Model is not None, "Model is None"
            model = self.Model(**best_params)


            model.fit(x_t, y_t)

            # 予測
            y_p = model.predict(x_v)
            # 精度の計算
            f1 = float(f1_score(y_v, y_p, average='macro'))
            logger.info(f"optuna phase {trial.number + 1} / 100 score: {f1}")
            return f1

        # Optunaでハイパーパラメータの最適化を行う
        # noinspection PyArgumentList
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100, n_jobs=2)

        # 最適なハイパーパラメータの表示
        best_params = study.best_params
        self.current_task = f"optuna done best params: {best_params}"

        return best_params

    def run(self):
        try:
            self.load()
            self.preprocess()
            self.k_fold_cross_validation()
            params = self.aggregate()
            logger.info(f'task finished')
            logger.info(f'{params.dict()}')
        except Exception as e:
            err_msg = f"{self.model.name} {self.ae.layers} error: {e}"
            logger.error(traceback.format_exc())
            logger.error(err_msg)
            self.send_error(err_msg)
        finally:
            return self

    def run_only_generate_encoder(self):
        try:
            self.load()
            self.preprocess()
            self.k_fold_cross_validation(only_generate_encoder=True)
        except Exception as e:
            err_msg = f"{self.model.name} {self.ae.layers} error: {e}"
            logger.error(traceback.format_exc())
            logger.error(err_msg)
            self.send_error(err_msg)
        finally:
            return self

    def __del__(self):
        logger.info(f"deleting {self.dataset.name} {self.model.name} {self.ae.layers} {self.ae.used_class}")
        keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        del self


if __name__ == '__main__':
    pass
