import pandas as pd

from utils_kdd99 import *


def main():
    X, y = load_data(use_full_dataset=False, standard_scale=True, verbose=0, )

    y = y.map(lambda x: attack_label_class[x]).map(lambda x: correspondences[x])
    split_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] = train_test_split(X, y,
                                                                                           test_size=0.33,
                                                                                           random_state=RANDOM_SEED,
                                                                                           stratify=y)
    x_train, x_test, y_train, y_test = split_data
    # x_train.drop(ignore_columns)
    x_train_dropped_column = x_train.drop(columns=ignore_columns)
    x_test_dropped_column = x_test.drop(columns=ignore_columns)

    print(x_train_dropped_column.head())
    x_train_dropped_column.to_pickle("models/kdd99_features/x_train_dropped_column_df.pkl")
    x_test_dropped_column.to_pickle("models/kdd99_features/x_test_dropped_column_df.pkl")

    """
    duration  src_bytes  ...  dst_host_rerror_rate  dst_host_srv_rerror_rate
    212221 -0.067792  -0.002017  ...              -0.25204                 -0.249464
    30903  -0.067792  -0.002774  ...              -0.25204                 -0.249464
    9739   -0.067792  -0.002017  ...              -0.25204                 -0.249464
    37540  -0.067792  -0.002776  ...              -0.25204                 -0.249464
    418638 -0.067792  -0.002535  ...              -0.25204                 -0.249464
    [5 rows x 26 columns]
    """

if __name__ == '__main__':
    main()
