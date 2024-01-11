from abc import ABC, abstractmethod
from datetime import datetime as dt, timezone, timedelta
import os
import traceback
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
import warnings
import tensorflow as tf
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
# from lightgbm import log_evaluation TODO

from general_utils import generate_encoder, insert_results
from notifier import LineClient
from tensorflow import keras
import optuna
from copy import deepcopy

VERSION = '1.6.0'

# logger.add('logs/base_flow.log', rotation='5 MB', retention='10 days', level='INFO')
ROOT_DIR = os.getcwd()
warnings.simplefilter('ignore')
logger.info(f"GPU {tf.config.list_physical_devices('GPU')}")
if not tf.config.list_physical_devices('GPU'):
    result = input("GPU could not be detected. Do you want to continue using CPU? [y/n]")
    if result == 'y' or result == 'Y':
        logger.warning("Running only on CPU")
    else:
        logger.error("The program was terminated because no GPU was found.")
        exit(1)


class BaseFlow(ABC):
    """
    BaseFlowは、機械学習の基本的なフローを抽象化したクラスです。

    Attributes:
        _current_task (str): 現在のタスクの状態を表す文字列。
        start_time (datetime): タスクの開始時間。
        config (dict): 設定情報を格納する辞書。
        Model: 使用するモデル。
        debug (bool): デバッグモードが有効かどうかを示すフラグ。
        random_seed (int): 乱数のシード値。
        splits (int): データの分割数。
        ae_used_data: (str): AutoEncoderが使用するデータ。
        model_name (str): モデルの名前。
        encoder_param (dict): エンコーダのパラメータ。
        _layers (list[int]): エンコーダのレイヤー情報。
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

    def __init__(self, **config) -> None:
        self._current_task: str = 'not_started'
        self.start_time: dt = dt.now(tz=timezone(timedelta(hours=9)))  # 実際には、runが呼ばれた時点での時刻
        self.config: dict = deepcopy(config)
        self.Model = None
        self.debug: bool = False if config['debug'] is None else self.config['debug']
        self.random_seed: int = self.config['random_seed']
        self.splits: int = self.config['splits']
        self.ae_used_data = self.config['ae_used_data']  # all or specific class
        self.model_name: str = self.config['model_name']
        self.encoder_param: dict = self.config['encoder_param']
        self._layers: list[int] = self.encoder_param['layers']
        self.model_param: dict = self.config['model_param']
        self.model_param['random_state'] = self.random_seed
        self.y_pred: Optional[pd.Series] = None
        self.x: Optional[pd.DataFrame] = None
        self.x_preprocessed: Optional[pd.DataFrame] = None
        self.x_new_features: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.encoder: Optional[keras.Sequential] = None
        self.conf_matrix: Optional[pd.DataFrame] = None
        self.scores = list()
        self.standard_scale = config.get('standard_scale', False)
        self.ae_standard_scale = config.get('ae_standard_scale', False)

        self.labels: Optional[list[str]] = None
        self.correspondence: Optional[dict[str, int]] = None
        self.output = {
            'random_seed': self.random_seed,
            'splits': self.splits,
            'model_name': self.model_name,
            'encoder_param': self.encoder_param,
            'model_param': self.model_param,
            'result': dict(),
            'importances': dict(),
            'standard_scale': self.standard_scale,
            'ae_standard_scale': self.ae_standard_scale,
        }
        assert type(self.config) is dict, f"config is {type(self.config)}"
        assert type(self.random_seed) is int, f"random_seed is {type(self.random_seed)}"
        assert type(self.splits) is int, f"splits is {type(self.splits)}"
        assert type(self.model_name) is str, f"model_name is {type(self.model_name)}"
        assert type(self.encoder_param) is dict, f"encoder_param is {type(self.encoder_param)}"
        assert type(self._layers) is list, f"config['layers'] is {type(self._layers)}"
        assert type(self.encoder_param['epochs']) is int
        assert type(self.encoder_param['batch_size']) is int
        assert type(self.encoder_param['activation']) is str

    @property
    def feature_size(self) -> int:
        if self.x is None:
            return 0
        if self._layers is None or self._layers == []:
            return self.x.shape[1]
        return self.x.shape[1] + self._layers[-1]

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
        logger.info(f'{self.name}: {self.model_name}{self._layers} {self.ae_used_data} started: {value}')

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
        assert self.x is not None, "x is None"
        #
        # self.x = pd.DataFrame(MinMaxScaler().fit_transform(self.x), columns=self.x.columns, index=self.x.index)
        # self.x = pd.DataFrame(StandardScaler().fit_transform(self.x), columns=self.x.columns, index=self.x.index)

    def _k_fold_preprocess(self, x_train, y_train, x_test, y_test) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        データの前処理を行います。

        このメソッドは、データの前処理を行います。
        データの前処理は、以下の手順で行われます。
        1. データの標準化
        2. エンコーダーの生成
        3. 新たな特徴量の生成

        Args:
            x_train (pd.DataFrame): 訓練データの入力データ。
            y_train (pd.Series): 訓練データの目的変数。
            x_test (pd.DataFrame): テストデータの入力データ。
            y_test (pd.Series): テストデータの目的変数。

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: 前処理済みの入力データ。

        """
        # データの標準化
        if self.standard_scale:
            scaler = StandardScaler()
            x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
            x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index) # type: ignore
        return x_train, x_test

    def _k_fold_generate_new_features(self, x_train, y_train, x_test, y_test) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        if self.ae_used_data == 'all':
            x_train_ae = x_train
        else:
            assert type(self.correspondence) is dict
            x_train_ae = x_train[y_train == self.correspondence[self.ae_used_data]]

        # モデルのファイル名
        option_str = "_d" if hasattr(self, "dropped") and self.dropped else "" # type: ignore
        option_str += "_ss" if self.ae_standard_scale else ""
        file_name = f"{self.name}{option_str}_{self._layers}_{self.ae_used_data}_{VERSION}.h5"

        # 保存済みモデルがある場合
        if os.path.exists(ROOT_DIR + "/models/" + file_name):
            # 保存しているモデルの読み込み
            _encoder = keras.models.load_model(ROOT_DIR + "/models/" + file_name)
        # 保存済みモデルがない場合
        else:
            self.current_task = f"train phase: encoder generating"
            _encoder = generate_encoder(x_train_ae, **self.encoder_param)
            # モデルの保存
            _encoder.save(ROOT_DIR + "/models/" + file_name)
        # 新たな特徴量を生成
        x_train_new_features = pd.DataFrame(
            _encoder.predict(x_train, verbose=0),  # type: ignore
            columns=[f"ae_{idx}" for idx in range(self._layers[-1])],
            index=x_train.index
        )
        x_test_new_features = pd.DataFrame(
            _encoder.predict(x_test, verbose=0),  # type: ignore
            columns=[f"ae_{idx}" for idx in range(self._layers[-1])],
            index=x_test.index
        )
        return x_train_new_features, x_test_new_features

    def _k_fold_hyperparameter_tuning(self, x_train, y_train) -> None:
        """
        ハイパーパラメータのチューニングを行います。

        このメソッドは、ハイパーパラメータのチューニングを行います。
        ハイパーパラメータのチューニングは、以下の手順で行われます。
        1. ハイパーパラメータのチューニング

        Args:
            x_train (pd.DataFrame): 訓練データの入力データ。
            y_train (pd.Series): 訓練データの目的変数。

        Returns:
            None

        """

        _names = self.model_name.split('+')
        if len(_names) == 2 and _names[1] == 'optuna':
            self.model_param = self.optuna(x_train, y_train)
            self.config['model_param'] = self.model_param

    def k_fold_cross_validation(self) -> None:
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
        self.start_time: dt = dt.now(tz=timezone(timedelta(hours=9)))
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

            if self._layers:
                # 新たな特徴量の生成
                x_train_new_features, x_test_new_features = self._k_fold_generate_new_features(x_train, y_train, x_test, y_test)

                # データを結合
                x_train = pd.concat([x_train, x_train_new_features], axis=1)
                x_test = pd.concat([x_test, x_test_new_features], axis=1)

            # ハイパーパラメータのチューニング
            self._k_fold_hyperparameter_tuning(x_train, y_train)

            # モデルの初期化
            assert type(self.Model) is not None, "Model is None"
            _model = self.Model(**self.model_param)

            # モデルを学習
            _model.fit(x_train, y_train)

            # 予測
            y_pred = pd.Series(_model.predict(x_test), index=y_test.index)

            # テストデータで評価
            accuracy: dict = classification_report(y_test, y_pred, output_dict=True)  # type: ignore

            if hasattr(_model, "feature_importances_"):
                for k, v in zip(x_test.columns, _model.feature_importances_):
                    if k in self.output["importances"]:
                        self.output["importances"][k] += int(v)
                    else:
                        self.output["importances"][k] = int(v)

            self.scores.append(accuracy)
            self.conf_matrix += confusion_matrix(y_test, y_pred, labels=range(len(self.labels)))

    def aggregate(self) -> None:
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
                    self.output['result'][s_labels[sl]][k2] = int(
                        np.sum([self.scores[i][sl][k2] for i in range(self.splits)]))
                else:
                    self.output['result'][s_labels[sl]][k2] = np.mean(
                        [self.scores[i][sl][k2] for i in range(self.splits)]).round(4)

        # 混同行列の集計
        confusion_dict = self.conf_matrix.T.rename(
            columns={idx: val for idx, val in enumerate(self.labels)},
            index={idx: val for idx, val in enumerate(self.labels)}
        ).to_dict()
        for true_label, pred_labels in confusion_dict.items():
            for pred_label, value in pred_labels.items():
                self.output['result'][true_label]["pred_" + pred_label] = value

        # 特徴量の数を出力に追加
        self.output['dataset']['total_feature'] = self.feature_size
        self.output['dataset']['default_feature'] = self.x.shape[1]  # type: ignore
        self.output['dataset']['ae_feature'] = self._layers[-1] if self._layers else 0

        self.output['datetime'] = self.datetime
        self.output['elapsed_time'] = str(self.elapsed_time)
        self.output['version'] = VERSION

        # 結果を出力
        if not self.debug:
            insert_results(self.output)

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
        if self.model_name == 'LogisticRegression+optuna':
            return dict(
                C = trial.suggest_loguniform('C', 0.001, 100),
                solver = trial.suggest_categorical('solver', ['lbfgs', 'sag']),
                multi_class = trial.suggest_categorical('multi_class', ['ovr', 'multinomial']),
            )
        elif self.model_name == 'SVM+optuna':
            return dict(
                C = trial.suggest_loguniform('C', 0.001, 100),
                kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
                gamma = trial.suggest_categorical('gamma', ['scale', 'auto']),
            )
        elif self.model_name == 'RandomForest+optuna':
            return dict(
                n_estimators = trial.suggest_int('n_estimators', 10, 1000),
                max_depth = trial.suggest_int('max_depth', 1, 100),
                min_samples_split = trial.suggest_int('min_samples_split', 2, 100),
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 100),
                max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            )
        elif self.model_name == 'MultiPerceptron+optuna':
            return dict(
                hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(15, 10, 5), (10, 5), (5,)]),
                activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu']),
                solver = trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam']),
                alpha = trial.suggest_loguniform('alpha', 0.0001, 1),
                learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
            )
        elif self.model_name == 'LightGBM+optuna':
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
                num_leaves=trial.suggest_int('num_leaves', 2, 200),
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
                verbosity=-1,
            )
        else:
            raise ValueError(f"model_name is {self.model_name}. It is not supported.")

    def optuna(self, x_train, y_train) -> dict:

        # 検証とテストに分割
        x_t, x_v, y_t, y_v = train_test_split(x_train, y_train, test_size=0.25, random_state=self.random_seed, stratify=y_train)

        def objective(trial):
            self.current_task = f'optuna phase {trial.number + 1} / 100'
            # ハイパーパラメータの設定
            params = self._get_default_params(trial)

            # モデルの学習
            assert self.Model is not None, "Model is None"
            model = self.Model(**params)
            model.fit(x_t, y_t)

            # 予測
            y_p = model.predict(x_v)
            # 精度の計算
            return float(f1_score(y_v, y_p, average='macro'))

        # Optunaでハイパーパラメータの最適化を行う
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)

        # 最適なハイパーパラメータの表示
        best_params = study.best_params
        self.current_task = f"optuna done best params: {best_params}"

        return best_params

    def run(self):
        try:
            self.load()
            self.preprocess()
            self.k_fold_cross_validation()
            self.aggregate()
            logger.info(f'task finished')
        except Exception as e:
            err_msg = f"{self.model_name} {self._layers} error: {e}"
            logger.error(err_msg)
            self.send_error(err_msg)
        finally:
            return self

    def __del__(self):
        logger.info(f"deleting {self.name} {self.model_name} {self._layers} {self.ae_used_data}")
        keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        del self


if __name__ == '__main__':
    pass
