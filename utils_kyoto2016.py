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

from keras.layers import Dense, Dropout

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
