import json
from db_query import *
import pandas as pd

# dataset = 'creditcardfraud'
def download_all():
    records = {}
    for dataset in datasets:
        records[dataset] = list(fetch_all_records({DATASET: dataset}))
    with open('logs/results.v.2.0.0.json', 'w') as f:
        json.dump(records, f, indent=4, default=str)

if __name__ == '__main__':
    download_all()