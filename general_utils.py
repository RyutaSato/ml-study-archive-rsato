import os
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from db_query import fetch_all_records
from tensorflow import keras
from keras.layers import Dense
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
uri = os.getenv('MongoDBURI')


def generate_encoder(x: pd.DataFrame, **config) -> keras.Sequential:
    """
    Keys:
        - layers: List of integers, where each integer represents the size of a hidden layer in the encoder.
        - epochs: Number of epochs to train the encoder for.
        - activation: Activation function to use for the encoder.
        - batch_size: Batch size to use for the encoder.
    """
    assert type(config['layers']) is list, f"config['layers'] is {type(config['layers'])}"
    assert type(config['activation']) is str
    assert type(config['epochs']) is int
    assert type(config['batch_size']) is int

    _layers = config['layers']
    activation = config['activation']

    _model = keras.Sequential(
        [
            Dense(_layers[0], activation=activation, input_shape=(x.shape[1],), name="encoder0"),
            *[
                Dense(hidden_layer_size, activation=activation, name=f"encoder{idx + 1}")
                for idx, hidden_layer_size in enumerate(_layers[1:])
            ],
            *[
                Dense(hidden_layer_size, activation=activation)
                for hidden_layer_size in _layers[-2::-1]
            ],
            Dense(x.shape[1], activation=activation),
        ]
    )
    _model.compile(optimizer="adam", loss="mse")
    _model.fit(x, x, epochs=config['epochs'],
               batch_size=config['batch_size'],
               verbose=0  # 0: silent, 1: progress bar, 2: one line per epoch, # type: ignore
               )
    _model = keras.Sequential(_model.layers[: len(_layers)])
    _model.compile(optimizer="adam", loss="mse")
    return _model


def insert_results(outputs: dict) -> None:
    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))

    # Send a ping to confirm a successful connection
    client.admin.command('ping')
    db = client.get_database('ml')
    assert db is not None, "db is None"
    collection = db.get_collection('results')
    assert collection is not None, "collection is None"

    _result = collection.insert_one(outputs)
    assert _result.acknowledged, "insertion failed"


def output_to_csv(name, keys) -> None:
    data_list = fetch_all_records({"dataset.name": name})

    def get_depper(data, key):
        if '.' in key:
            return get_depper(data[key.split('.')[0]], '.'.join(key.split('.')[1:]))
        else:
            return data[key]

    # 対象のデータを抽出して新しい辞書を作成
    extracted_data = [{key: get_depper(data, key) for key in keys} for data in
                      data_list]

    # 集計用のデータフレームを作成
    df = pd.DataFrame(extracted_data)

    # 結果をcsvファイルに出力
    df.to_csv(os.getcwd() + f'/logs/{name}.csv', index=False)


def fit_and_predict(_x,
                    _y,
                    _Model,
                    n_splits,
                    random_seed,
                    **params) -> list[dict]:
    """
    Args:
        _x: Input data.
        _y: Target data.
        _Model: Model to use for prediction.
        n_splits: Number of folds to use for cross-validation.
        random_seed: Random state to use for cross-validation.
        **params: Parameters for prediction.

    """
    assert type(n_splits) is int
    assert type(random_seed) is int

    # k分割
    k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    _generator = k_fold.split(_x, _y)
    accuracies = []
    for fold, (train_idx, test_idx) in enumerate(_generator):
        logger.info(f"phase: {fold + 1}/{n_splits}")
        # データを分割
        x_train = _x.iloc[train_idx]
        y_train = _y.iloc[train_idx]
        x_test = _x.iloc[test_idx]
        y_test = _y.iloc[test_idx]
        _model = _Model(**params)
        # モデルを学習
        _model.fit(x_train, y_train)
        # テストデータで評価
        accuracy: dict = classification_report(y_test, _model.predict(x_test), output_dict=True)  # type: ignore
        accuracies.append(accuracy)
        logger.info(f"f1-score: {accuracy['macro avg']['f1-score']}")
    return accuracies


if __name__ == '__main__':
    output_to_csv('kdd99',
                  ['_id', 'datetime', 'dataset.name',
                   'model_name', 'dataset.dropped', 'dataset.ae_used_data', 'encoder_param.layers',
                   'result.u2r.f1-score',
                   'result.macro avg.f1-score'])
    # output_to_csv('creditcardfraud', ['_id', 'datetime', 'dataset.name', 'dataset.ae_used_data' , 'model_name',
    # 'encoder_param.layers', 'result.anomaly.f1-score', 'result.macro avg.f1-score'])
