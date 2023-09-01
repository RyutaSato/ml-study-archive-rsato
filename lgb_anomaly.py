import pickle

import pandas as pd

from utils_kdd99 import *


def main():
    print_version()
    # load data
    with open("models/kdd99_features/x_train_df.pkl", 'rb') as f:
        x_train: pd.DataFrame = pickle.load(f)
    with open("models/kdd99_features/y_train_df.pkl", 'rb') as f:
        y_train: pd.Series = pickle.load(f)
    # 正常ラベルのみをdrop
    y_train_anomaly = y_train[y_train != 1]

    x_train_anomaly = x_train[y_train != 1]
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 5,
        # :Warning:
        # LightGBMのモデルは，ラベルが連番でなければいけないので，ラベル1は除いたが，あるものとして学習させる．
        # そのため，実際にはクラス数は4であるが，5として扱う．

        'metric': 'multi_error',
        'learning_rate': 0.1,
        'num_leaves': 23,
        'min_data_in_leaf': 1,
        'verbose': -1,
        'random_state': RANDOM_SEED,
    }
    lgb_train = lgb.Dataset(x_train_anomaly, y_train_anomaly)
    print("train start")
    model = lgb.train(params,  # パラメータ
                      lgb_train,  # トレーニングデータの指定
                      valid_sets=[lgb_train],  # 検証データの指定
                      callbacks=[lgb.early_stopping(50, verbose=False)],
                      )
    print("train done")
    model.save_model('models/lightgbm/lgb_anomaly_tuned_booster.model')
    with open('models/lightgbm/lgb_anomaly_tuned_booster.pkl', 'wb') as fp:
        pickle.dump(model.dump_model(), fp)

if __name__ == '__main__':
    main()
