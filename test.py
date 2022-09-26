import os
import unittest
import logging
import datetime

from unittest import TestCase
from pybehavior.models import DataSet

from pybehavior.tools import Loader, Preprocessor

import pandas as pd

class WalkTestCase(TestCase):
    def setUp(self):
        logging.basicConfig(filename='.log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')
        logging.debug('Starting Test: {}'.format(datetime.datetime.now()))
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_columns', None)  

    def try_to_load_data(self, data_id:int, src_data=None):
        data_dir = 'data/walk/'
        data_path = os.path.join(data_dir, "data{}.pickle".format(data_id))

        if os.path.exists(data_path):
            return pd.read_pickle(data_path)
        else:
            if data_id == 1:
                data = pd.read_csv('data/walk/jawbone.csv', low_memory=False)
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

    def test_sequence(self):
        data = self.try_to_load_data(1)
        data2 = self.try_to_load_data(2, data)
        data3 = self.try_to_load_data(3, data2)
        data4 = self.try_to_load_data(4, data3)
        data5 = self.try_to_load_data(5, data4)
        data6 = self.try_to_load_data(6, data5)
        data7 = self.try_to_load_data(7, data6)
        logging.debug(data7)
        
class EMATestCase(TestCase):
    def setUp(self):
        logging.basicConfig(filename='.log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')
        logging.debug('Starting Test: {}'.format(datetime.datetime.now()))
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_columns', None)  

    def try_to_load_data(self, data_id:int, src_data=None):
        data_dir = 'data/ema/'
        data_path = os.path.join(data_dir, "data{}.pickle".format(data_id))

        if os.path.exists(data_path):
            return pd.read_pickle(data_path)
        else:
            if data_id == 1:
                data = pd.read_csv('data/ema/ema.csv', low_memory=False)
            elif data_id == 2:
                data = src_data.copy()
                data['question_id'] = self.recode_question_id(src_data)
            elif data_id == 3:
                data = src_data[['rnum', 'created', 'question_id', 'answer_value']]
                data['created'] = pd.to_datetime(data['created']).dt.date
            elif data_id == 4:
                data = src_data.groupby(['rnum', 'created', 'question_id']).agg({"answer_value": "mean"})
                data = data.pivot_table(index=['rnum', 'created'], columns=['question_id'], values='answer_value')
                data = data.sort_values(['rnum', 'created']).reset_index()
                data.columns.name = None
            elif data_id == 5:
                data = Preprocessor.fill_missing_dates(src_data, id='rnum', date='created', columns=[
                        'E5',
                        'E6',
                        'E7',
                        'E8',
                        'E10',
                        'E11',
                        'E12.1.1',
                        'E12.1.2',
                        'E12.4',
                        'E12.7',
                        'E13.1',
                        'E13.2',
                        'E13.3',
                        'E13.4',
                        'E13.5',
                        'E13.6',
                        'E13.7',
                        'E14'
                    ]
                )
            else:
                raise Exception("Unknown data_id: {}".format(data_id))
            data.to_pickle(data_path)
            return data

    def recode_question_id(self, src_data):
        question_label = [
            'Being active is a <b>top priority</b> tomorrow. ',
            '<b>Circumstances will help me</b> to be active tomorrow (e.g., nice weather, getting in nature, free time).',
            'My <b>schedule makes it easy</b> to be active tomorrow.',
            'I <b>expect obstacles</b> (e.g., no time, unsafe, poor weather) to being active tomorrow.',
            'I know how to <b>solve any problems</b> to being active tomorrow.',
            'I am confident I can <b>overcome obstacles</b> to being active tomorrow.',
            "<b>No matter what</b>, I'm going to be active tomorrow.",
            'In general, my <b>friends help me</b> to be active.',
            'I regularly feel <b>urges to</b> be active.',
            'I am active because it <b>helps me feel better</b> (e.g., reduce stress, stiffness, or fatigue).',
            'I have a <b>wide range of strategies</b> (e.g., call friends while walking) that I use to be active regularly. ',
            'My <b>typical Monday includes being active.</b>',
            'My <b>typical Tuesday includes being active.</b>',
            'My <b>typical Wednesday includes being active.</b>',
            'My <b>typical Thursday includes being active.</b>',
            'My <b>typical Friday includes being active.</b>',
            'My <b>typical Saturday includes being active.</b>',
            'My <b>typical Sunday includes being active.</b>'
        ]
        question_id = [
            'E5',
            'E7',
            'E6',
            'E8',
            'E12.1.1',
            'E10',
            'E11',
            'E14',
            'E12.7',
            'E12.4',
            'E12.1.2',
            'E13.1',
            'E13.2',
            'E13.3',
            'E13.4',
            'E13.5',
            'E13.6',
            'E13.7'
        ]

        return src_data['question_label'].replace(question_label, question_id)

    def test_all(self):
        data = self.try_to_load_data(1)
        data = self.try_to_load_data(2, data)
        data = self.try_to_load_data(3, data)
        data = self.try_to_load_data(4, data)
        data = self.try_to_load_data(5, data)
        logging.debug(data.shape)
        logging.debug(data)

if __name__ == '__main__':
    unittest.main()