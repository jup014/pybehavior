import datetime
import logging
import pandas as pd
from pandas.core.frame import DataFrame

class Participant:
    def __init__(self):
        print("Participant is constructed")


class DataSet:
    def __init__(self):
        pass
    
    def group_by_sum(self, by=datetime.timedelta(hours=1)) -> 'DataSet':
        participant_list = self.data['participant'].unique()
        
        headers = ['participant', 'start_time_local', 'end_time_local', 'value', 'timespan']
        new_data = []

        for each_participant in participant_list:
            each_participant_data = self.data[self.data['participant']==each_participant]

            each_participant_data.sort_values(by=['start_time_local', 'end_time_local'])
            
            first_start_time = each_participant_data.iloc[0].start_time_local
            first_day_midnight = datetime.datetime(first_start_time.year, first_start_time.month, first_start_time.day)

            last_end_time = each_participant_data.iloc[-1].end_time_local
            last_day_midnight = datetime.datetime(last_end_time.year, last_end_time.month, last_end_time.day) + datetime.timedelta(days=1)

            current_start_time = first_day_midnight
            current_end_time = current_start_time + by

            cumulative_value = 0

            for index, each_row in each_participant_data.iterrows():
                while current_end_time <= each_row['start_time_local']:
                    new_data.append([
                        each_participant, current_start_time, current_end_time, cumulative_value, by
                    ])
                    current_start_time += by
                    current_end_time += by
                    cumulative_value = 0

                if each_row['start_time_local'] < current_end_time:
                    cumulative_value += each_row['value']

            while current_end_time <= last_day_midnight:
                new_data.append([
                        each_participant, current_start_time, current_end_time, cumulative_value, by
                    ])
                current_start_time += by
                current_end_time += by
                cumulative_value = 0

        new_dataframe = pd.DataFrame(new_data, columns=headers)
        new_dataset = DataSet()
        new_dataset.data = new_dataframe

        return new_dataset
            
