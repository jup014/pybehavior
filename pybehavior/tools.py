import logging
import pandas as pd
from pandas import read_csv, to_datetime
from .models import DataSet



class Loader:
    def read_csv(filename,
        participant='user',
        start_time_local='start_utime_local',
        end_time_local='end_utime_local',
        value='steps'
    ) -> DataSet:
        pd.options.mode.chained_assignment = None  # default='warn'
        
        df = read_csv(filename, low_memory=False)

        data_set = DataSet()        
        df2 = df[[participant, start_time_local, end_time_local, value]]

        df2.columns.values[0] = 'participant'
        df2.columns.values[1] = 'start_time_local'
        df2.columns.values[2] = 'end_time_local'
        df2.columns.values[3] = 'value'

        df2['start_time_local'] = to_datetime(df2['start_time_local'])
        df2['end_time_local'] = df2['end_time_local'].astype('datetime64[ns]')

        df2['timespan'] = df2['end_time_local'] - df2['start_time_local']
        data_set.data = df2

        logging.debug(data_set.data.head())

        return data_set