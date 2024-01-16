from db_query import *
from datetime import datetime, timedelta

class Result:
    def __init__(
            self, 
            version: str, 
            model_name: str, 
            dataset_name: str,
            datetime: datetime,
            elapsed_time: timedelta,
            layers: list[int],
            ae_feature_num: int,
            base_feature_num: int,
            total_feature_num: int,
            ae_used_class: str,
            class_num: int,
            dataset_num: int,




    
    @classmethod
    def from_dict(cls, conditions: dict):
        result = fetch_latest_record(conditions)
        return cls(

        )