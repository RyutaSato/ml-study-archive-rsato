import os

from pymongo.mongo_client import MongoClient
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
uri = os.getenv('MongoDBURI')

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
client.admin.command('ping')
db = client.get_database('ml')
assert db is not None, "db is None"
collection = db.get_collection('results')
assert collection is not None, "collection is None"


def get_result(conditions: dict) -> dict:
    # 条件に一致するものの中で、もっとも新しいデータを１つ得る
    result = collection.find_one(conditions, sort=[('_id', -1)])
    return result


def get_results(conditions: dict) -> dict:
    results = collection.find(conditions)
    return results


if __name__ == '__main__':
    r = get_result({
        "dataset": {
            "name": "kdd99",
            "use_full": False,
            "dropped": False,
            "ae_used_data": "all"
        },
        'model_name': 'LogisticRegression',
    })
    logger.info(r)

    r = get_result({
        "dataset.name": "kdd99",
        'model_name': 'LogisticRegression',
    })
    logger.info(r)
