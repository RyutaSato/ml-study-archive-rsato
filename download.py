import json
from db_query import *

# dataset = 'creditcardfraud'
def download_all():
    records = fetch_all_records({})
    with open('results/results.json', 'w') as f:
        json.dump(records, f, default=str)

if __name__ == '__main__':
    download_all()