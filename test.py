import unittest
import logging
import datetime

from unittest import TestCase

from pybehavior.tools import Loader, Preprocessor

import pandas as pd

class ParticipantTestCase(TestCase):
    def setUp(self):
        logging.basicConfig(filename='.log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')
        logging.debug('Starting Test: {}'.format(datetime.datetime.now()))
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_columns', None)  
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
    
    def test_filter_out_2(self):
        data2 = Loader.pick_columns(data=self.data, user='user', start_datetime='start_utime_local', end_datetime='end_utime_local', values=['steps'])
        data3 = data2.query('steps >= 60')
        data4 = Preprocessor.merge_rows(data3)
        data5 = data4.query('duration >= datetime.timedelta(minutes=5)')

    def test_sample(self):
        data2 = Loader.pick_columns(data=self.data, user='user', start_datetime='start_utime_local', end_datetime='end_utime_local', values=['steps'])
        data3 = data2.query('steps >= 60')
        data4 = Preprocessor.merge_rows(data3)
        data5 = data4.query('duration >= datetime.timedelta(minutes=5)')

        hours_with_012 = Preprocessor.get_hourly_activity_data(data5)
        hours_with_012.to_csv('hours_with_012.csv')


if __name__ == '__main__':
    unittest.main()