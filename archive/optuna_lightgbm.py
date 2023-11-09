from utils_kdd99 import *
import optuna.integration.lightgbm as lgb

print_version()

"""
Warnings:
    this module was deprecated.
"""
def main():
    print_version()
    # load data
    with open("models/kdd99_features/x_train_df.pkl", 'rb') as f:
        x_train: pd.DataFrame = pickle.load(f)
    with open("models/kdd99_features/x_test_df.pkl", 'rb') as f:
        x_test: pd.DataFrame = pickle.load(f)
    with open("models/kdd99_features/y_train_df.pkl", 'rb') as f:
        y_train: pd.DataFrame = pickle.load(f)
    with open("models/kdd99_features/y_test_df.pkl", 'rb') as f:
        y_test: pd.DataFrame = pickle.load(f)
    # normal = 0, abnormal = 1
    # 正常ラベルと異常ラベルに変換する関数を定義
    convert_to_binary = lambda x: 0 if x == 1 else 1
    y_train_b = y_train.apply(convert_to_binary)
    y_test_b = y_test.apply(convert_to_binary)
    lgb_train = lgb.Dataset(x_train, y_train_b)
    # LightGBM parameters
    params = {
        'objective': 'binary',  # 2値分類の目的関数
        'metric': 'binary_logloss',  # 評価指標（対数損失）
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 23,
        'verbose': -1,
        'random_state': RANDOM_SEED,
    }
    # モデルの学習
    model = lgb.train(params,  # パラメータ
                      lgb_train,  # トレーニングデータの指定
                      valid_sets=[lgb_train],  # 検証データの指定
                      callbacks=[lgb.early_stopping(50, verbose=False)],
                      )

    y_pred_b = model.predict(x_test)
    y_pred_b = np.round(y_pred_b)  # 予測確率を0か1に変換
    print(y_pred_b)
    print(classification_report(y_test_b, y_pred_b))
    print(model.params)
    model.save_model("models/lightgbm/lgb_binary_tuned_booster.model")
    with open("models/lightgbm/lgb_binary_param_tuned_booster.pkl", 'wb') as fp:
        pickle.dump(model.dump_model(), fp)


if __name__ == '__main__':
    main()
