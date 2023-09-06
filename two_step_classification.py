from utils_kdd99 import *


# 第一段階　2値分類
def classification_normal_and_anomaly(X: pd.DataFrame, model: LogisticRegression | lgb.Booster) -> pd.Series:
    y_pred = model.predict(X)
    y_pred = np.round(y_pred)
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    y_pred = pd.Series(y_pred, index=X.index, )
    return y_pred


# 第二段階　異常データの分類
def classification_anomalies(X: pd.DataFrame, model: LogisticRegression | lgb.Booster) -> pd.Series:
    y_pred = model.predict(X)
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)  # 一番大きい予測確率のクラスを予測クラスに
    y_pred = pd.Series(y_pred, index=X.index)
    return y_pred


def two_step_classification(X, y, model_1st, model_2nd, verbose=2, return_value='cm'):
    """二段階分類用の関数
    1段階目：正常と異常の2値分類を行う．
    二段階目：異常と判断されたデータのみを抽出し，4つの異常ラベルに分類する．

    Args:
        X (pd.DataFrame):
        y (pd.Series):
        model_1st (LogisticRegression | lgb.Booster): 2値分類モデル
        model_2nd (LogisticRegression | lgb.Booster): 異常クラス分類モデル
        verbose (int):
            2の場合: 混合行列および，分類レポートを出力する．
            1の場合: 混合行列を出力する．
            0の場合: 何も出力しない．
        return_value (Literal['cm', 'predict']):
            'cm': returns confusion matrix
            'predict': returns predicted Y
    Returns:
        (pd.DataFrame, pd.DataFrame) | (pd.Series, pd.Series):
            'cm'：1段階目と，2段階目の混合行列のタプルを返す．
            'predict'：1段階目と，2段階目の予測のタプルを返す．
    """
    y_pred_binary = classification_normal_and_anomaly(X, model_1st)
    predicted_indexes = y_pred_binary[y_pred_binary == 1].index
    x_anomalies = X.loc[predicted_indexes]
    y_pred_anomalies: pd.Series = classification_anomalies(x_anomalies, model_2nd)
    print(y_pred_anomalies.value_counts())
    y_pred_normal = y_pred_binary[y_pred_binary == 0].apply(lambda _: 1)
    y_pred = pd.concat([y_pred_normal, y_pred_anomalies])

    y_test_binary = y.apply(lambda x: 0 if x == 1 else 1)

    cm_1st = confusion_matrix_df(y_test_binary.sort_index(), y_pred_binary.sort_index(), labels=['normal', 'anomaly'])

    cm_2nd = confusion_matrix_df(y.sort_index(), y_pred.sort_index())

    if verbose > 1:
        print(classification_report(y_test_binary, y_pred_binary))
        print(classification_report(y.sort_index(), y_pred.sort_index(), target_names=correspondences.keys()))
    if verbose > 0:
        print(cm_1st)
        print(cm_2nd)
    if return_value == 'cm':
        return cm_1st, cm_2nd
    elif return_value == 'predict':
        return y_pred_binary, y_pred_anomalies
