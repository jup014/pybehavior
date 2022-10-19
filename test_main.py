import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import BayesLDM.BayesLDM as BayesLDM
from pybehavior.tools import MCMC_Parser, BayesModelBuilder
from tinydb import TinyDB

data:pd.DataFrame = pd.read_pickle('data/ema/data11.pickle')
data = data.rename(columns={'steps': 'behavior', 'cue_to_action': 'cue'})

rnum = 17

data = data.loc[data.rnum == rnum].reset_index(drop=True)

# ['rnum', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12.1.1',
#        'E12.1.2', 'E12.4', 'E12.5', 'E12.6', 'E12.7', 'day_index',
#        'day_of_week', 'steps', 'AWND', 'PRCP', 'TMAX', 'TMIN', 'self_efficacy',
#        'perceived_barrier', 'environmental_context', 'context_of_walking',
#        'typicalness_of_context', 'self_management_skills', 'cue_to_action',
#        'environmental_context_weekly', 'social_support_friends']

data.name = "data"

builder = BayesModelBuilder('model')

builder.add_variable_regression_edge([
    ('self_efficacy', 'self_efficacy'),
    # ('perceived_barrier', 'perceived_barrier'),
    # ('perceived_barrier', 'self_efficacy'),
    # ('behavior', 'behavior'),
    # ('cue', 'cue'),
    # ('self_efficacy', 'behavior'),
    # ('cue', 'behavior')
    ])

# print(builder.get_full_model())

N = data.shape[0]

db = TinyDB('data/ema/db.json')

columns = [
            'self_efficacy', 
            # 'perceived_barrier', 
            # 'cue', 
            # 'behavior'
            ]

for window_size in range(10, N):
    actual_window_size = window_size + 1        # including the last missing data
    for start_point in range(0, N - actual_window_size):
        print("cropping {}:{}".format(start_point, actual_window_size + start_point + 1))
        current_data = data.iloc[start_point:(start_point + actual_window_size) + 1]
        original_data = current_data.loc[actual_window_size:actual_window_size]
        current_data.loc[actual_window_size:actual_window_size, columns] = None  # type: ignore
        current_data = current_data.assign(t=range(0, 12))
        builder.set_index('t', 0, actual_window_size)

        model = BayesLDM.compile(builder.get_full_model(), obs=columns, data=[data]) 
        samples = model.sample(b_post_process=True)
        parser = MCMC_Parser(model)

        distance = 0
        estimated = []
        for each_column in columns:
            each_estimated = int(parser.variable[each_column][actual_window_size])
            estimated.append(each_estimated)
            distance += pow(original_data[each_column] - each_estimated, 2)
        distance = math.sqrt(distance) / len(columns)  # type: ignore

        db.insert({'rnum': rnum, 'window_size': window_size, 'columns': columns, 'start_point': start_point, 'distance': distance, 'original_data': original_data[columns].iloc[0].to_list(), 'estimated': estimated})






# print(data.loc[10:10])
