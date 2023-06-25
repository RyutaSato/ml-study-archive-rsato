import platform
import sklearn
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import tomli
import lightgbm as lgb

from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc, log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt

from keras.src.layers import Dense, Dropout
from keras.src import regularizers


def print_version():
    print(f"python:      {platform.python_version()} (text: 3.6)")
    print(f"sklearn:     {sklearn.__version__}")
    print(f"tensorflow:  {tf.__version__} (text: 1.14)")
    print(f"keras:       {keras.__version__}")
    print(f"numpy:       {np.__version__}")
    print(f"pandas:      {pd.__version__}")


def load_config(section_name: str):
    sections: list = section_name.split('.')
    with open("config.toml", "rb+") as f:
        toml = tomli.load(f)
    for section in sections:
        toml = toml[section]
    return toml


def load_credit_card_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config("default")
    data = pd.read_csv(cfg['dataset_path'])
    data_x = data.copy().drop(cfg['drop_x'], axis=1)
    data_y = data[cfg['label_y']].copy()
    return data_x, data_y


def standard_scale(x: pd.DataFrame) -> pd.DataFrame:
    # `StandardScaler`は，全ての特徴量の平均を0，標準偏差を1になるようにする．
    # これにより，異なる範囲の特徴量を持つデータを正規化する．
    from sklearn.preprocessing import StandardScaler
    return pd.DataFrame(StandardScaler().fit_transform(x))


def plot_results(true_labels, anomaly_scores_, return_preds=False):
    preds = pd.concat([true_labels, anomaly_scores_], axis=1)
    preds.columns = ['true_label', 'anomaly_score']
    precision, recall, thresholds = \
        precision_recall_curve(preds['true_label'], preds['anomaly_score'])
    average_precision = average_precision_score(
        preds['true_label'], preds['anomaly_score']
    )
    plt.step(recall, precision, color='k', alpha=0.7, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    plt.title('Precision-Recall curve: Average Precision = '
              '{0:0.2f}'.format(average_precision))

    fpr, tpr, thresholds = \
        roc_curve(preds['true_label'], preds['anomaly_score'])

    area_under_roc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.xlabel([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: Area under the curve = {0:0.2f}'.format(area_under_roc))
    plt.legend(loc='lower right')
    plt.show()

    if return_preds:
        return preds, average_precision


def anomaly_scores(original_df: pd.DataFrame, reconstructed_df: pd.DataFrame, reversed=False) -> np.ndarray:
    """
    元の特徴量行列と，新たに再構成された特徴量行列の間の再構成誤差を計算する．
    各特徴量ごとの再構成前後の誤差を二乗し，全ての特徴量を足し合わせる．正規化して0〜1に納める．
    0に近いほど正常，１に近いほど異常
    :param original_df: 元の特徴量の行列
    :param reconstructed_df: 再構成された特徴量の行列
    :param reversed:
    :return:
    """
    loss: np.ndarray = np.sum((np.array(original_df) - np.array(reconstructed_df)) ** 2, axis=1)
    loss: pd.Series = pd.Series(loss, index=original_df.index)
    loss: np.ndarray = (loss - np.min(loss)) / (np.max(loss) - np.min(loss))
    if reversed:
        loss = 1 - loss
    return loss


def precision_analysis(df: pd.DataFrame, column, threshold: float):
    """
    あるレベルの再現率(Recall)に対する適合率(Precision)を求める．
    再現率 = 真陽性 / (真陽性 + 偽陰性)
    適合率 = 真陽性　/ (真陽性 + 偽陽性)
    不正ラベルのうち，実際に検出できた不正の割合（再現率）が
    :param df:
    :param column:
    :param threshold:
    :return:
    Example:
        threshold = 0.4の場合，40％の再現率を達成した際の適合率
    """
    df.sort_values(by=column, ascending=False, inplace=True)
    threshold_value = threshold * df['true_label'].sum()  # 不正ラベルの個数
    i = 0
    j = 0
    while i <= threshold_value:
        if df.iloc[j]["true_label"] == 1:
            i += 1
        j += 1
    return df, i / j


def precision_for_given_recall(df: pd.DataFrame, column, recall_percentage: float):
    """
     あるレベルの再現率(Recall)に対する適合率(Precision)を求める．
    再現率 = 真陽性 / (真陽性 + 偽陰性)
    適合率 = 真陽性　/ (真陽性 + 偽陽性)
    不正ラベルのうち，実際に検出できた不正の割合（再現率）が`recall_percentage`である場合の真と予測した内本当に真であった割合（適合率）
    :param df:
    :param column:
    :param recall_percentage:
    :return:
    """
    return precision_analysis(df, column, recall_percentage)[1]


def reconstruction_errors(original_df: pd.DataFrame, reconstructed_df: pd.DataFrame) -> np.ndarray:
    """
    元の特徴量行列と，新たに再構成された特徴量行列の間の再構成誤差を計算する．
    :param original_df:
    :param reconstructed_df:
    :return:
    """
    return anomaly_scores(original_df, reconstructed_df)


if __name__ == '__main__':
    load_credit_card_dataset()
    print(load_config('supervised.k_fold'))
