import os
from typing import Optional

from pymongo.mongo_client import MongoClient
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
_uri = os.getenv('MongoDBURI')

# Create a new client and connect to the server
_client = MongoClient(_uri)

# Send a ping to confirm a successful connection
_client.admin.command('ping')
_db = _client.get_database('ml')
assert _db is not None, "db is None"
_collection = _db.get_collection('results')
assert _collection is not None, "collection is None"

versions = ["1.0.0", "1.1.0", "1.2.0", "1.2.1", "1.2.2", "1.2.3", "1.2.4", "1.2.5"]
LATEST = versions[-1]  # 現在の最新バージョン
DATASET = "dataset.name"
MODEL = "model_name"



def fetch_latest_record(conditions: dict) -> Optional[dict]:
    # 条件に一致するものの中で、もっとも新しいデータを１つ得る
    result = _collection.find_one(conditions, sort=[('_id', -1)])
    return result


def fetch_all_records(conditions: dict):
    results = _collection.find(conditions)
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
