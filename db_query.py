import os
from typing import Optional

from pymongo.mongo_client import MongoClient
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
# <<<<<<< features/refact_que
# =======
# _uri = os.getenv('MongoDBURI')

# # Create a new client and connect to the server
# _client = MongoClient(_uri)

# # Send a ping to confirm a successful connection
# _client.admin.command('ping')
# _db = _client.get_database('ml')
# assert _db is not None, "db is None"
# _collection = _db.get_collection('results_v.2.0.0') # version<2.0.0 -> `results`
# assert _collection is not None, "collection is None"
# >>>>>>> main

versions = ["1.0.0", "1.1.0", "1.2.0", "1.2.1", "1.2.2", "1.2.3", "1.2.4", "1.2.5", "1.3.0", "1.3.1", "1.4.0", "1.5.0", "1.5.1", "2.0.0"]

datasets = ['kdd99', 'kdd99_dropped', 'creditcardfraud', 'ecoli', 'optical_digits', 'satimage', 'pen_digits', 'abalone', 'sick_euthyroid', 'spectrometer', 'car_eval_34', 'isolet', 'us_crime', 'yeast_ml8', 'scene', 'libras_move', 'thyroid_sick', 'coil_2000', 'arrhythmia', 'solar_flare_m0', 'oil', 'car_eval_4', 'wine_quality', 'letter_img', 'yeast_me2', 'webpage', 'ozone_level', 'mammography', 'protein_homo', 'abalone_19']
LATEST = versions[-1]  # 現在の最新バージョン
DATASET = "dataset.name"
_uri = os.getenv('MongoDBURI')

def get_collection(version: str):
    if version < "2.0.0":
        collection_s = 'results'
    else:
        collection_s = 'results_v.' + version
    _client = MongoClient(_uri)
    _client.admin.command('ping')
    _db = _client.get_database('ml')
    assert _db is not None, "db is None"
    _collection = _db.get_collection(collection_s)
    assert _collection is not None, "collection is None"
    return _collection
MODEL = "model.name" # version<2.0.0 -> `model_name`


_collection = get_collection("2.0.0")


def fetch_latest_record(conditions: dict) -> Optional[dict]:
    # 条件に一致するものの中で、もっとも新しいデータを１つ得る
    result = _collection.find_one(conditions, sort=[('_id', -1)])
    return result


def fetch_all_records(conditions: dict):
    results = _collection.find(conditions)
    return list(results)


def done_experiment(h: str) -> bool:
    try:
        value = _collection.find_one({"hash": h}) is not None
    except Exception:
        _collection = get_collection("2.0.0")
        value = _collection.find_one({"hash": h}) is not None
        globals()['_collection'] = _collection
    return value


def done_experiments() -> set:
    try:
        hashes = _collection.find({}, {"_id": 0, "hash": 1})
    except Exception:
        _collection = get_collection("2.0.0")
        hashes = [res["hash"] for res in _collection.find({}, {"_id": 0, "hash": 1})]
        globals()['_collection'] = _collection
    return set(hashes)


if __name__ == '__main__':
    print(done_experiment("bm9uZVswXWxya2RkOTltYWpvcml0eUZhbHNl"))
    print(done_experiment("bm9uZVswXWxya2RkOTlhbGxGYWxzZQ="))
    r = fetch_latest_record({
        "dataset": {
            "name": "kdd99",
            "use_full": False,
            "dropped": False,
            "ae_used_data": "all"
        },
        'model_name': 'LogisticRegression',
    })
    logger.info(r)

    r = fetch_latest_record({
        "dataset.name": "kdd99",
        'model_name': 'LogisticRegression',
    })
    logger.info(r)
