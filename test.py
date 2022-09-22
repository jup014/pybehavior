import os
import unittest
import logging
import datetime

from unittest import TestCase
from pybehavior.models import DataSet

from pybehavior.tools import Loader, Preprocessor

import pandas as pd

class ParticipantTestCase(TestCase):
    def setUp(self):
        logging.basicConfig(filename='.log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')
        logging.debug('Starting Test: {}'.format(datetime.datetime.now()))
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_columns', None)  

    def try_to_load_data(self, data_id:int, src_data=None):
        data_dir = 'data/'
        data_path = os.path.join(data_dir, "data{}.pickle".format(data_id))

        if os.path.exists(data_path):
            return pd.read_pickle(data_path)
        else:
            if data_id == 1:
                data = pd.read_csv('data/jawbone.csv', low_memory=False)
            elif data_id == 2:
                data = Loader.pick_columns(data=src_data, user='user', start_datetime='start_utime_local', end_datetime='end_utime_local', values=['steps'])
            elif data_id == 3:
                data = src_data.query('steps >= 60')
            elif data_id == 4:
                data = Preprocessor.merge_rows(src_data)
            elif data_id == 5:
                data = src_data.query('duration >= datetime.timedelta(minutes=5)')
            elif data_id == 6:
                data = Preprocessor.get_hourly_activity_data(src_data)
            elif data_id == 7:
                dataset = DataSet(src_data)
                today = datetime.datetime(2015, 8, 15)
                data = dataset.query(user=1, start=today - datetime.timedelta(days=35), end=today, values=['activity'])
            else:
                raise Exception("Unknown data_id: {}".format(data_id))
            data.to_pickle(data_path)
            return data

    def test_pick_columns(self):
        data = self.try_to_load_data(1)
        data2 = self.try_to_load_data(2, data)
    
    def test_filter_out_1(self):
        data = self.try_to_load_data(1)
        data2 = self.try_to_load_data(2, data)
        data3 = self.try_to_load_data(3, data2)

    def test_merge_rows(self):
        data = self.try_to_load_data(1)
        data2 = self.try_to_load_data(2, data)
        data3 = self.try_to_load_data(3, data2)
        data4 = self.try_to_load_data(4, data3)
    
    def test_filter_out_2(self):
        data = self.try_to_load_data(1)
        data2 = self.try_to_load_data(2, data)
        data3 = self.try_to_load_data(3, data2)
        data4 = self.try_to_load_data(4, data3)
        data5 = self.try_to_load_data(5, data4)

    def test_sample(self):
        data = self.try_to_load_data(1)
        data2 = self.try_to_load_data(2, data)
        data3 = self.try_to_load_data(3, data2)
        data4 = self.try_to_load_data(4, data3)
        data5 = self.try_to_load_data(5, data4)
        data6 = self.try_to_load_data(6, data5)

    def test_query(self):
        data = self.try_to_load_data(1)
        data2 = self.try_to_load_data(2, data)
        data3 = self.try_to_load_data(3, data2)
        data4 = self.try_to_load_data(4, data3)
        data5 = self.try_to_load_data(5, data4)
        data6 = self.try_to_load_data(6, data5)
        data7 = self.try_to_load_data(7, data6)
        logging.debug(data7)
        

if __name__ == '__main__':
    unittest.main()