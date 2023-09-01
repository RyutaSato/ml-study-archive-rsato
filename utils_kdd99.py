import platform
import sklearn
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import tomli
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import optuna
import pickle

from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc, log_loss, \
    accuracy_score, classification_report, multilabel_confusion_matrix, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


try:
    # colab環境上
    from google.colab import drive

    drive.mount('/content/drive')
    pwd = '/content/drive/MyDrive/dataset/'
    is_colab = True
    from keras.src.layers import Dense, Dropout
    from keras.src import regularizers
except ImportError:
    import os

    # ローカル環境
    pwd = os.getcwd() + '/dataset/'
    is_colab = False
    from keras.layers import Dense, Dropout
    from keras import regularizers

RANDOM_SEED = 2018


def print_version():
    print(f"python:      {platform.python_version()}")
    print(f"sklearn:     {sklearn.__version__}")
    print(f"tensorflow:  {tf.__version__}")
    print(f"keras:       {keras.__version__}")
    print(f"numpy:       {np.__version__}")
    print(f"pandas:      {pd.__version__}")


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
    print(thresholds)
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


# anomaly_score関数を改名したもの
def reconstruction_errors(original_df: pd.DataFrame, reconstructed_df: pd.DataFrame) -> np.ndarray:
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
    return loss


def load_data(use_full_dataset=False, standard_scale=True, verbose=1):
    """
    データの読み込み
    :return:
    """
    # 特徴量名の読み込み
    with open(pwd + "kddcup.names") as fp:
        # 一行目は不要なので無視
        _ = fp.readline()
        # `:`より手前がラベルなので，その部分を抽出してリストに追加
        names = [line.split(':')[0] for line in fp]
    if verbose:
        print(f"特徴量の数：{len(names)}")
        print(f"各特徴量の名前：{', '.join(names)}")
    # 　正解ラベルを追加
    names.append("true_label")
    if use_full_dataset:
        data = pd.read_csv(pwd + "kddcup.data", names=names, index_col=False)
    else:
        data = pd.read_csv(pwd + "kddcup.data_10_percent", names=names, index_col=False)
    data_x = data.copy()
    data_x = data_x.drop(columns=['protocol_type', 'service', 'flag'], axis=1)
    true_label = data_x.pop('true_label')  # 正解ラベルのピリオドを外す．
    names = data_x.columns
    if standard_scale:
        from sklearn.preprocessing import StandardScaler
        data_x = StandardScaler().fit_transform(data_x)
        data_x = pd.DataFrame(data_x, columns=names)
    true_label = true_label.map(lambda x: x.replace('.', ''))
    if verbose:
        print(true_label.value_counts())
        print(data_x.shape)
    return data_x, true_label


# attack_class_labels -> key: class, value: list[label]
attack_class_labels = {
    'normal': ['normal'],
    'dos': ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop'],
    'u2r': ['buffer_overflow', 'loadmodule', 'perl', 'rootkit'],
    'r2l': ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster'],
    'probe': ['ipsweep', 'nmap', 'portsweep', 'satan']
}

# class -> int
correspondences = {
    'dos': 0,
    'normal': 1,
    'probe': 2,
    'r2l': 3,
    'u2r': 4
}

# attack_class_label -> key: label, value: class
attack_label_class = {}
for c, labels in attack_class_labels.items():
    for label in labels:
        attack_label_class[label] = c

def confusion_matrix_df(y_true, y_pred, labels=correspondences.keys()):
    return pd.DataFrame(confusion_matrix(y_true, y_pred),
                        index=["true_" + label for label in labels],
                        columns=labels)
