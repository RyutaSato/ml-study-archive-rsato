import os
from typing import Optional

from pymongo.mongo_client import MongoClient
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

versions = ["1.0.0", "1.1.0", "1.2.0", "1.2.1", "1.2.2", "1.2.3", "1.2.4", "1.2.5", "1.3.0", "1.3.1", "1.4.0", "1.5.0"]
LATEST = versions[-1]  # 現在の最新バージョン
DATASET = "dataset.name"
MODEL = "model_name"


def get_collection(version: str):
    if version < "2.0.0":
        collection_s = 'results'
    else:
        collection_s = 'results_v.' + version
    _uri = os.getenv('MongoDBURI')
    _client = MongoClient(_uri)
    _client.admin.command('ping')
    _db = _client.get_database('ml')
    assert _db is not None, "db is None"
    _collection = _db.get_collection('results')
    assert _collection is not None, "collection is None"
    return _collection


def fetch_latest_record(conditions: dict) -> Optional[dict]:
    # 条件に一致するものの中で、もっとも新しいデータを１つ得る
    result = get_collection('2.0.0').find_one(conditions, sort=[('_id', -1)])
    return result


def fetch_all_records(conditions: dict):
    results = get_collection('2.0.0').find(conditions)
    return results


if __name__ == '__main__':
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
