import os.path
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from utils_kdd99 import *
import logging

logger = logging.getLogger('LR')
logging.basicConfig(level=logging.INFO)

def main():
    x_train = pd.read_pickle('models/kdd99_features/x_train+ae_43_df&activation=relu&epochs=5&batch_size=32.pkl')
    x_test = pd.read_pickle('models/kdd99_features/x_test+ae_43_df&activation=relu&epochs=5&batch_size=32.pkl')
    y_train = pd.read_pickle('models/kdd99_features/y_train_df.pkl')
    y_test = pd.read_pickle('models/kdd99_features/y_test_df.pkl')
    # customize parameters
    parameters = {
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
        'random_state': RANDOM_SEED,
        'C': [10 ** i for i in range(-5, 6)]
    }
    for penalty in ['l1', 'l2']:
        for solver in ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']:
            for C in [10 ** i for i in range(-5, 2)]:
                param_str = f"models/logistic_regression/kdd99+ae_43&penalty={penalty}&solver={solver}&C={C}.pkl"
                if os.path.isfile(param_str):
                    logger.info("skipped: " + param_str)
                    continue
                logger.info("start: " + param_str)
                try:
                    model = LogisticRegression(penalty=penalty, C=C, solver=solver, n_jobs=12)
                    model.fit(x_train, y_train)
                except ValueError as e:
                    logger.error("skipped: " + param_str)
                    logger.error(e)
                    continue
                pickle.dump(model, open(param_str, 'wb'))
                logger.info("saved: " + param_str)
    logger.info("competed")


if __name__ == '__main__':
    main()
