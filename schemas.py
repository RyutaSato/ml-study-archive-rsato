from datetime import timedelta, datetime

from pydantic import BaseModel


class GeneralWorkLoad(BaseModel):
    preprocess: list[str] = []
    layers: list[list[int]] = []
    models: list[str] = []
    datasets: list[str] = []
    ae_used_class: list[str] = []
    optuna: list[bool] = [False]


class IndividualWorkLoad(BaseModel):
    preprocess: str = ""
    layers: list[int] = []
    model: str = ""
    dataset: str = ""
    ae_used_class: str = ""
    optuna: bool = False


class WorkLoad(BaseModel):
    general: GeneralWorkLoad = GeneralWorkLoad()
    individual: list[IndividualWorkLoad] = []


class Environment(BaseModel):
    version: str
    datetime: datetime
    elapsed_time: timedelta = timedelta(seconds=0)


class Dataset(BaseModel):
    name: str
    default_feature_num: int = 0
    ae_feature_num: int = 0
    total_feature_num: int = 0
    sample_num: int = 0
    standardization: bool = False
    normalization: bool = False


class Accuracy(BaseModel):
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0


class Result(BaseModel):
    majority: Accuracy = Accuracy()
    minority: Accuracy = Accuracy()
    macro: Accuracy = Accuracy()
    importances: dict = {}


class MLModel(BaseModel):
    name: str
    optuna: bool = False
    params: dict = {}
    best_params_list: list[dict] = []


class AEModel(BaseModel):
    layers: list[int] = [0]
    used_class: str = "all"  # "majority", "minority", "all"
    epochs: int = 10
    activation: str = "relu"
    batch_size: int = 32
    standardization: bool = False
    normalization: bool = False


class Params(BaseModel):
    hash: str
    dataset: Dataset
    model: MLModel
    ae: AEModel
    env: Environment
    result: Result


if __name__ == '__main__':
    ae = AEModel()
    print(ae.dict(), id(ae.layers))
    ae.layers.append(3)
    print(ae.dict(), id(ae.layers))
