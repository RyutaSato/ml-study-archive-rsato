
from utils_kdd99 import *
import optuna.integration.lightgbm as lgb

def main():
    print_version()
    # # load data
    # with open("models/kdd99_features/x_train_df.pkl", 'rb') as f:
    #     x_train: pd.DataFrame = pickle.load(f)
    size = 40
    with open(f"models/kdd99_features/x_train+ae_48_df&activation=relu&epochs=5&batch_size=32.pkl", 'rb') as f:
        x_train: pd.DataFrame = pickle.load(f)
    with open("models/kdd99_features/y_train_df.pkl", 'rb') as f:
        y_train: pd.Series = pickle.load(f)
    x_train_binary = x_train
    # Binaryラベルに変換
    y_train_binary = y_train.apply(lambda x: 0 if x == 1 else 1)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        # :Warning:
        # LightGBMのモデルは，ラベルが連番でなければいけないので，ラベル1は除いたが，あるものとして学習させる．
        # そのため，実際にはクラス数は4であるが，5として扱う．

        'metric': 'auc',
        'learning_rate': 0.1,
        'num_leaves': 23,
        'min_data_in_leaf': 1,
        'verbose': -1,
        'random_state': RANDOM_SEED,
    }
    lgb_train = lgb.Dataset(x_train_binary, y_train_binary)
    print("train start")
    model = lgb.train(params,  # パラメータ
                      lgb_train,  # トレーニングデータの指定
                      valid_sets=[lgb_train],  # 検証データの指定
                      callbacks=[lgb.early_stopping(50, verbose=False)],
                      )
    print("train done")
    # model.save_model(f'models/lightgbm/lgb_dropped+ae_{size}_binary_tuned_booster.model')
    model.save_model(f'models/lightgbm/lgb+ae_48_binary_tuned_booster.model')

    # with open('models/lightgbm/lgb_dropped__25_binary_tuned_booster.pkl', 'wb') as fp:
    #     pickle.dump(model.dump_model(), fp)

if __name__ == '__main__':
    main()
