import unittest
import logging
import datetime

from unittest import TestCase

from pybehavior.models import Participant, DataSet, Preprocessor
from pybehavior.tools import Loader

import pandas as pd

class ParticipantTestCase(TestCase):
    def setUp(self):
        logging.basicConfig(filename='.log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')
        logging.debug('Starting Test: {}'.format(datetime.datetime.now()))
        self.data = pd.read_csv('data/jawbone.csv', low_memory=False)

    def test_pick_columns(self):
        data2 = Loader.pick_columns(data=self.data, user='user', start_datetime='start_utime_local', end_datetime='end_utime_local', values=['steps'])
    
    def test_filter_out_1(self):
        data2 = Loader.pick_columns(data=self.data, user='user', start_datetime='start_utime_local', end_datetime='end_utime_local', values=['steps'])
        data3 = data2.query('steps >= 60')

    def test_merge_rows(self):
        data2 = Loader.pick_columns(data=self.data, user='user', start_datetime='start_utime_local', end_datetime='end_utime_local', values=['steps'])
        data3 = data2.query('steps >= 60')
        data4 = Preprocessor.merge_rows(data3)
        


    
    # def test_get_continuous_behavior(self):
    #     data = Loader.read_csv('data/jawbone.csv')
    #     data = data.filter_out(60, mode="lt")
    #     data = data.get_continuous_behavior()

    # def test_filter_out_2(self):
    #     data = Loader.read_csv('data/jawbone.csv')
    #     data = data.filter_out(60, mode="lt")
    #     data = data.get_continuous_behavior()
    #     data = data.filter_out(datetime.timedelta(minutes=5), mode="lt", column="timespan")
    
    # def test_group_by_sum(self):
    #     data = Loader.read_csv('data/jawbone.csv')
    #     data = data.filter_out(60, mode="lt")
    #     data = data.get_continuous_behavior()
    #     data = data.filter_out(datetime.timedelta(minutes=5), mode="lt", column="timespan")
    #     data = data.group_by_sum()

if __name__ == '__main__':
    unittest.main()