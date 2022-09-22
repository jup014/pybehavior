import datetime
import logging
from typing import List
import pandas as pd
from pandas import read_csv, to_datetime
from pandas.core.frame import DataFrame



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


class Preprocessor:
    def merge_rows(data: DataFrame):
        data = data.reset_index()
        data['shifted_end'] = data.groupby(['user'])[['end']].shift(fill_value=datetime.datetime(datetime.MAXYEAR, 12, 31, 23, 59))
        data['group'] = ((1 - (data['start'] == data['shifted_end'])).cumsum())
        data = data.groupby(['user', 'group']).agg({'start': 'first', 'end': 'last', 'steps': 'mean'}).reset_index()
        logging.debug(data.head(10))
        data['duration'] = data['end'] - data['start']
        data = data.set_index(['user', 'start', 'end']).drop('group', axis=1)

        return data
    
    def get_resampled_data(data:pd.DataFrame, group:str=None, start='start', end='end', span_unit='hour', unit='hour'):
        if group:
            data = data[[group, start, end]].drop_duplicates()
        else:
            data = data[[start, end]].drop_duplicates()

        if span_unit == 'hour':
            data['starthour'] = pd.to_datetime(data[start].dt.date) + pd.to_timedelta(data[start].dt.hour, unit='hour')
            data['endhour'] = pd.to_datetime(data[end].dt.date) + pd.to_timedelta(data[end].dt.hour + 1, unit='hour')
        elif span_unit == 'day':
            data['starthour'] = pd.to_datetime(data[start].dt.date)
            data['endhour'] = pd.to_datetime(data[end].dt.date) + datetime.timedelta(days=1)

        if group:
            data = data[[group, 'starthour', 'endhour']].drop_duplicates()
        else:
            data = data[['starthour', 'endhour']].drop_duplicates()
        
        def t(row):
            if group:
                group_info = getattr(row, group)
            
            start_datetime = row.starthour
            end_datetime = row.endhour

            timedelta = (end_datetime - start_datetime).total_seconds()

            if unit == 'hour':
                timedelta = timedelta / 3600
                timefreq = 'H'
            elif unit == 'day':
                timedelta = timedelta / 86400
                timefreq = 'D'
            
            date_range = pd.date_range(start_datetime, periods=timedelta, freq=timefreq)
            data_dict = {
                    start: date_range,
                    end: date_range + datetime.timedelta(hours=1)
                }
            if group:
                data_dict[group] = group_info
            return pd.DataFrame(data_dict)
        
        data = pd.concat(data.apply(t, axis=1).tolist()).reset_index(drop=True)
        if group:
            column_order = [group, 'start', 'end']
        else:
            column_order = ['start', 'end']
        data = data[column_order].drop_duplicates().sort_values(column_order).reset_index(drop=True)

        return data

    def get_hourly_activity_data(data):
        data = data.reset_index()
        hours_with_activity = Preprocessor.get_resampled_data(data=data, group='user', start='start', end='end', span_unit='hour', unit='hour')
        hours_with_activity['activity'] = 2

        hours_with_data = Preprocessor.get_resampled_data(data=data, group='user', start='start', end='end', span_unit='day', unit='hour')
        hours_with_12 = pd.merge(hours_with_activity, hours_with_data, on=['user', 'start', 'end'], how="outer").fillna(1)
        
        full_day_range = hours_with_data.groupby('user').agg({'start': min, 'end': max}).reset_index()
        
        hours_full_day_range = Preprocessor.get_resampled_data(full_day_range, group='user', start='start', end='end', span_unit='day', unit='hour')
        
        hours_with_012 = pd.merge(hours_with_12, hours_full_day_range, on=['user', 'start', 'end'], how='outer').fillna(0)
        hours_with_012['activity'] = hours_with_012['activity'].astype(int)
        hours_with_012 = hours_with_012.sort_values(['user', 'start', 'end']).reset_index(drop=True)
        
        return hours_with_012