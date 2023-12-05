from db_query import *
import pandas as pd

DATA = 'creditcardfraud'
def main():
    records = list(fetch_all_records({DATASET: DATA}))
    df = pd.json_normalize(records)
    df.to_csv(f"logs/{DATA}.csv")

if __name__ == '__main__':
    main()