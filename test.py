import unittest
import logging
import datetime

from unittest import TestCase

from pybehavior.models import Participant, DataSet
from pybehavior.tools import Loader

class ParticipantTestCase(TestCase):
    def setUp(self):
        logging.basicConfig(filename='.log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')
        logging.debug('Starting Test: {}'.format(datetime.datetime.now()))
        
    def test_load_data(self):
        data = Loader.\
            read_csv('data/jawbone.csv').\
                filter_out(60, mode="lt").\
                    get_continuous_behavior().\
                        filter_out(datetime.timedelta(minutes=5), mode="lt", column="timespan").\
                            group_by_sum()

if __name__ == '__main__':
    unittest.main()