import datetime
import logging
import pandas as pd
from pandas.core.frame import DataFrame

from pybehavior.tools import Preprocessor

class DataSet:
    def __init__(self, data):
        self.data = data

    def query(self, user, start, end, values):
        df = DataFrame(
            {
                'user': [user],
                'start': [start],
                'end': [end]
            }
        )
        empty_hours = Preprocessor.get_resampled_data(df, 'user')
        q = self.data.query('(user == @user) and (start >= @start) and (end <= @end)').sort_values('start')
        q2 = pd.merge(empty_hours, q, on=['user', 'start', 'end'], how="outer").fillna(0)
        
        return q2['activity']


class EMA(DataSet):
    def __init__(self, data, user='user', label_column='question_label'):
        super().__init__(data)

        self.item_refresh()

