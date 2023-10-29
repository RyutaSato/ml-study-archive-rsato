import pandas as pd

from utils_kdd99 import *


def main():
    X, y = load_data(use_full_dataset=False, standard_scale=True, verbose=0, )

    y = y.map(lambda x: attack_label_class[x]).map(lambda x: correspondences[x])
    split_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] = train_test_split(X, y, test_size=0.33, random_state=RANDOM_SEED, stratify=y)
    x_train, x_test, y_train, y_test = split_data
    # x_train.drop(ignore_columns)
    x_train_dropped_column = x_train.drop(columns=ignore_columns)
    x_test_dropped_column = x_test.drop(columns=ignore_columns)

    print(x_train_dropped_column.shape)
    # x_train_dropped_column.to_pickle("models/kdd99_features/x_train_dropped_column_df.pkl")
    # x_test_dropped_column.to_pickle("models/kdd99_features/x_test_dropped_column_df.pkl")

if __name__ == '__main__':
    main()
