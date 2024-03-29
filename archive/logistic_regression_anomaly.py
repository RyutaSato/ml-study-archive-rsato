import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from utils_kdd99 import *
import logging

logger = logging.getLogger('LR')
logging.basicConfig(level=logging.INFO)

def main():
    # input train and test data
    # X, y = load_data(use_full_dataset=False, standard_scale=True, verbose=0, )
    # # convert X labels to numbers
    # y = y.map(lambda x: attack_label_class[x]).map(lambda x: correspondences[x])
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RANDOM_SEED, stratify=y)

    x_train: pd.DataFrame = pd.read_pickle("models/kdd99_features/x_train_dropped_anomaly_df.pkl")
    y_train: pd.Series = pd.read_pickle("models/kdd99_features/y_train_dropped_mapped_series.pkl")
    #
    # y_train_anomaly = y_train[y_train != 1]
    # x_train_anomaly = x_train[y_train != 1]
    logger.info(f"x_train shape: {x_train.shape}, y_train len: {len(y_train)}")
    logger.info(f"y_train value_counts: \n{y_train.value_counts()}")

    for penalty in ['l1', 'l2']:
        # for solver in ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']:
        for solver in ['liblinear']:
            # for C in [10 ** i for i in range(-5, 2)]:
            for C in [10 ** -5, 10 ** -1, 1, 10]:
                param_str = f"models/logistic_regression_anomaly/kdd99-dropped_mapped&penalty={penalty}&solver" \
                            f"={solver}&C" \
                            f"={C}.pkl"
                # if os.path.isfile(param_str):
                #     logger.info("skipped: " + param_str)
                logger.info("start: " + param_str)
                try:
                    model = LogisticRegression(penalty=penalty, C=C, solver=solver, n_jobs=12)
                    # model.fit(x_train_anomaly, y_train_anomaly)
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
