import datetime
import logging
import pandas as pd

class Participant:
    def __init__(self):
        print("Participant is constructed")


class DataSet:
    def __init__(self):
        pass
    
    def filter_out(self, cutoff, mode='lt', column='value') -> 'DataSet':
        if mode == 'lt':
            temp_data = self.data[self.data[column] >= cutoff]
        elif mode == 'lte':
            temp_data = self.data[self.data[column] > cutoff]
        elif mode == 'gt':
            temp_data = self.data[self.data[column] <= cutoff]
        elif mode == 'gte':
            temp_data = self.data[self.data[column] < cutoff]
        else:
            raise Exception("Invalid mode: {}".format(mode))
        
        new_dataset = DataSet()
        new_dataset.data = temp_data

        return new_dataset

    def get_continuous_behavior(self) -> 'DataSet':
        participant_list = self.data['participant'].unique()
        
        headers = ['participant', 'start_time_local', 'end_time_local', 'value', 'timespan']
        new_data = []

        for each_participant in participant_list:
            each_participant_data = self.data[self.data['participant']==each_participant]

            each_participant_data.sort_values(by=['start_time_local', 'end_time_local'])
            
            current_start_time = each_participant_data.iloc[0]['start_time_local']
            current_end_time = each_participant_data.iloc[0]['end_time_local']
            cumulative_value = each_participant_data.iloc[0]['value']

            for index, each_row in each_participant_data.iterrows():
                each_row_start_time = each_row['start_time_local']
                each_row_end_time = each_row['end_time_local']

                if each_row_start_time == current_start_time:
                    cumulative_value += each_row['value']
                else:
                    if each_row_start_time == current_end_time:
                        current_end_time = each_row_end_time
                        cumulative_value += each_row['value']
                    else:
                        new_row = [each_participant, current_start_time, current_end_time, cumulative_value, current_end_time - current_start_time]
                        new_data.append(new_row)
                        current_start_time = each_row_start_time
                        current_end_time = each_row_end_time
                        cumulative_value = each_row['value']
        
        new_dataframe = pd.DataFrame(new_data, columns=headers)
        new_dataset = DataSet()
        new_dataset.data = new_dataframe

        return new_dataset
    
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
            
