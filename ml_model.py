
from abc import ABC, abstractmethod
import pandas as pd


class ModelBase(ABC):

    @abstractmethod
    def load(self) -> None:
        pass
    
    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def predict(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def save(self) -> None:
        pass

    

