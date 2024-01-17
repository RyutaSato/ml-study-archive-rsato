import json
from db_query import *
import pandas as pd

# dataset = 'creditcardfraud'
def download_all():
    records = fetch_all_records({})
    with open('../logs/results.v.2.0.0.json', 'w') as f:
        json.dump(records, f, default=str)

if __name__ == '__main__':
    download_all()