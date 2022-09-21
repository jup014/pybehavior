import logging
from typing import List
import pandas as pd
from pandas import read_csv, to_datetime
from pandas.core.frame import DataFrame
from .models import DataSet



class Loader:
    def pick_columns(
        data:DataFrame,
        user:str,
        start_datetime:str,
        end_datetime:str,
        values:List[str]
    ) -> DataFrame:
        column_list = [user, start_datetime, end_datetime] + values
        return_data = data[column_list].rename(columns={user: 'user', start_datetime: 'start', end_datetime: 'end'})
        return_data['start'] = pd.to_datetime(return_data.start)
        return_data['end'] = pd.to_datetime(return_data.end)
        return_data = return_data.set_index(['user', 'start', 'end'])
        return return_data