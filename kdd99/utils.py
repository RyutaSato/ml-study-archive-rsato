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
    print(f"python:      {platform.python_version()}")
    print(f"sklearn:     {sklearn.__version__}")
    print(f"tensorflow:  {tf.__version__}")
    print(f"keras:       {keras.__version__}")
    print(f"numpy:       {np.__version__}")
    print(f"pandas:      {pd.__version__}")