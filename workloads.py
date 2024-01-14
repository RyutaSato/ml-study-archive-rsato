import json
from datetime import timedelta, datetime
from itertools import product

import requests
import yaml
import base64

from schemas import Params, Environment, Dataset, Accuracy, Result, MLModel, AEModel, WorkLoad

URL = 'http://localhost:8080'


def load_workloads() -> WorkLoad:
    with open('workloads.yml', 'r') as f:
        cfg = WorkLoad(**yaml.safe_load(f))
    return cfg


def gen_hash(*args) -> str:
    return base64.b64encode("".join([str(arg) for arg in args]).encode()).decode()


def main():
    # 前回失敗した実験を再実行する
    with open('results/not_finished.json', 'r') as f:
        data = json.load(f)
        print("not finished: ", len(data))
        for params_str in data:
            params = Params(**json.loads(params_str))
            r = requests.post(URL + '/in_queue', data=params.json())
            print(r.json())

    workloads: WorkLoad = load_workloads()
    itr = product(
        workloads.general.preprocess,
        workloads.general.layers,
        workloads.general.models,
        workloads.general.datasets,
        workloads.general.ae_used_class,
        workloads.general.optuna,
    )
    itr = list(itr)

    for individual in workloads.individual:
        print(individual)
        _preprocess = [individual.preprocess] if individual.preprocess else workloads.general.preprocess
        _layers = [individual.layers] if individual.layers else workloads.general.layers
        _models = [individual.model] if individual.model else workloads.general.models
        _datasets = [individual.dataset] if individual.dataset else workloads.general.datasets
        _ae_used_class = [individual.ae_used_class] if individual.ae_used_class else workloads.general.ae_used_class
        _optuna = [individual.optuna] if individual.optuna else workloads.general.optuna
        itr.extend(product(_preprocess, _layers, _models, _datasets, _ae_used_class, _optuna))
    print(len(itr))
    for preprocess, layers, model, dataset, used_class, optuna in itr:
        standardization, normalization, ae_standardization, ae_normalization = False, False, False, False
        if preprocess == 'ae_standardization':
            standardization, ae_standardization = True, True
        elif preprocess == 'ae_normalization':
            normalization, ae_normalization = True, True
        elif preprocess == 'standardization':
            standardization = True
        elif preprocess == 'normalization':
            normalization = True

        params = Params(
            hash=gen_hash(preprocess, layers, model, dataset, used_class, optuna),
            dataset=Dataset(name=dataset, standardization=standardization, normalization=normalization),
            model=MLModel(name=model, optuna=optuna),
            ae=AEModel(layers=layers, used_class=used_class, standardization=ae_standardization,
                       normalization=ae_normalization),
            env=Environment(version='2.0.0', datetime=datetime.now(), elapsed_time=timedelta(seconds=0), ),
            result=Result(),
        )
        print(params)
        r = requests.post(URL + '/in_queue', data=params.json())
        print(r.json())


if __name__ == '__main__':
    main()
