import os
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

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
    assert config['layers'] is list
    assert config['activation'] is str
    assert config['epochs'] is int
    assert config['batch_size'] is int

    layers = config['layers']
    activation = config['activation']

    _model = keras.Sequential( 
        [
            Dense(layers[0], activation=activation, input_shape=(x.shape[1],), name="encoder0"),
            *[
                Dense(hidden_layer_size, activation=activation, name=f"encoder{idx + 1}")
                for idx, hidden_layer_size in enumerate(layers[1:])
            ],
            *[
                Dense(hidden_layer_size, activation=activation)
                for hidden_layer_size in layers[-2::-1]
            ],
            Dense(x.shape[1], activation=activation),
        ]
    )
    _model.summary()
    _model.compile(optimizer="adam", loss="mse")
    _model.fit(x, x, epochs=config['epochs'], batch_size=config['batch_size'])
    return keras.Sequential(_model.layers[: len(layers)])


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
    if _result is None:
        print("successfully reflected results to DB")
    else:
        print("failed to reflect results to DB")

def predict(_x, _y, _model, **params) -> list[dict]:
    """
    Args:
        _x: Input data.
        _y: Target data.
        _model: Model to use for prediction.
        **params: Parameters for prediction.

    Keys:
        - n_splits: Number of folds to use for cross-validation.
        - random_state: Random state to use for cross-validation.

    """
    assert params['n_splits'] is int
    assert params['random_state'] is int

    # k分割
    k_fold = StratifiedKFold(n_splits=params['n_splits'], shuffle=True, random_state=params['random_state'])
    _generator = k_fold.split(_x, _y)
    accuracies = []
    for fold, (train_idx, test_idx) in enumerate(_generator):
        print(f"fold: {fold}")
        # データを分割
        x_train = _x.iloc[train_idx]
        y_train = _y.iloc[train_idx]
        x_test = _x.iloc[test_idx]
        y_test = _y.iloc[test_idx]

        # モデルを学習
        _model.fit(x_train, y_train)
        # テストデータで評価
        accuracy = classification_report(y_test, _model.predict(x_test), output_dict=True)
        accuracies.append(accuracy)
        print(f"f1-score: {accuracy['macro avg']['f1-score']}") # type: ignore
    return accuracies