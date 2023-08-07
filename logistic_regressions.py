import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from utils_kdd99 import *
import logging

logger = logging.getLogger('LR')
logging.basicConfig(level=logging.INFO)

def main():
    # input train and test data
    X, y = load_data(use_full_dataset=False, standard_scale=True, verbose=0, )
    # 分類モデルによっては数値ラベルしか対応していないため、目的変数を分類クラスから数値へ変換
    y = y.map(lambda x: attack_label_class[x]).map(lambda x: correspondences[x])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RANDOM_SEED, stratify=y)
    # customize parameters
    parameters = {
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
        'random_state': RANDOM_SEED,
        'C': [10 ** i for i in range(-5, 6)]
    }
    for penalty in ['l1', 'l2']:
        for solver in ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']:
            for C in [10 ** i for i in range(-5, 6)]:
                param_str = f"models/logistic_regression/kdd99_38&penalty={penalty}&solver={solver}&C={C}.pkl"
                logger.info("start: " + param_str)
                model = LogisticRegression(penalty, C=C, solver=solver)
                model.fit(x_train, y_train)
                pickle.dump(model, open(param_str, 'wb'))
                logger.info("saved: " + param_str)
    logger.info("competed")


if __name__ == '__main__':
    main()
