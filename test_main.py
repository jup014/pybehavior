import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import BayesLDM.BayesLDM as BayesLDM
from pybehavior.tools import MCMC_Parser, BayesModelBuilder
from tinydb import TinyDB, Query


relations = [
    ('self_efficacy', 'self_efficacy'),
    ('perceived_barrier', 'perceived_barrier'),
    ('perceived_barrier', 'self_efficacy'),
    ('behavior', 'behavior'),
    ('cue', 'cue'),
    ('self_efficacy', 'behavior'),
    ('cue', 'behavior')
    ]

columns = list(set([x for (x, y) in relations] + [y for (x, y) in relations]))
columns2 = columns.copy()
columns2.remove('behavior')

data_original:pd.DataFrame = pd.read_pickle('data/ema/data11.pickle')
data_original = data_original.rename(columns={'steps': 'behavior', 'cue_to_action': 'cue'})

rnum_list = [4, 5, 6, 8, 9, 11, 17, 18, 22, 25, 26, 27, 30, 31, 33, 34, 42, 43, 44, 47, 48]

db = TinyDB('data/ema/db.json')
queue = TinyDB('data/ema/queue.json')

if not os.path.exists('data/ema/queue.json'):
    for rnum in rnum_list:
        data = data_original.loc[data_original.rnum == rnum].reset_index(drop=True).copy()
        data = data[columns]

        data.name = "data"
        
        N = data.shape[0]

        save_queue = []
        for window_size in range(10, N):
            actual_window_size = window_size + 1        # including the last missing data
            for start_point in range(0, N - actual_window_size - 1):
                save_queue.append({'rnum': rnum, 'window_size': window_size, 'start_point': start_point, 'status': 'queued'})
                if len(save_queue) == 1000:
                    queue.insert_multiple(save_queue)
                    del save_queue
                    save_queue = []
        del data

        queue.insert_multiple(save_queue)
        del save_queue
        save_queue = []

while True:
    Job = Query()

    job = queue.get(Job.status == 'queued')
    if job is None:
        break
    
    queue.update({'status': 'fetched'}, (Job.rnum == job['rnum']) & (Job.window_size == job['window_size']) & (Job.start_point == job['start_point']))

    rnum = job['rnum']
    window_size = job['window_size']
    start_point = job['start_point']

    data = data_original.loc[data_original.rnum == rnum].reset_index(drop=True).copy()
    data = data[columns]

    data.name = "data"

    N = data.shape[0]

    actual_window_size = window_size + 1        # including the last missing data
    
    Log = Query()
    log_count = db.count((Log.rnum == rnum) & (Log.method == 'Sustain') & (Log.window_size == window_size) & (Log.start_point == start_point))
    if log_count == 0:
        print("cropping {}:{}".format(start_point, actual_window_size + start_point))
        current_data = data.loc[start_point:(start_point + window_size)].reset_index(drop=True).copy()
        original_data = current_data.loc[window_size:window_size].copy()
        if original_data.isna().sum(axis=1).to_list()[0] > 0:
            db.insert({'rnum': rnum, 'method': 'BayesLDM', 'window_size': window_size, 'columns': columns, 'start_point': start_point, 'distance': None, 'original_data': original_data[columns].iloc[0].to_list(), 'estimated': None})
            db.insert({'rnum': rnum, 'method': 'Sustain', 'window_size': window_size, 'columns': columns, 'start_point': start_point, 'distance': None, 'original_data': original_data[columns].iloc[0].to_list(), 'estimated': None})
        else:
            builder = BayesModelBuilder('model')
            builder.add_variable_regression_edge(relations)
            # bayes LDM method
            current_data.loc[window_size:window_size, columns] = None  # type: ignore
            current_data = current_data.assign(t=range(0, actual_window_size))
            builder.set_index('t', 0, window_size)
            current_data.name = 'dataset'
            model = BayesLDM.compile(builder.get_full_model(), obs=columns, data=[current_data]) 
            samples = model.sample(b_post_process=True)
            parser = MCMC_Parser(model)

            distance = 0
            estimated = []
            
            for each_column in columns2:
                each_estimated = int(parser.variable[each_column][window_size])
                estimated.append(each_estimated)
                distance += pow(original_data.iloc[0][each_column] - each_estimated, 2)
            
            distance = math.sqrt(distance) / len(columns2)  # type: ignore

            db.insert({'rnum': rnum, 'method': 'BayesLDM', 'window_size': window_size, 'columns': columns2, 'start_point': start_point, 'distance': None if distance is np.NaN else distance, 'original_data': original_data[columns].iloc[0].to_list(), 'estimated': estimated})

            # sustain method
            distance = 0
            estimated = []
            for each_column in columns2:
                a_col = current_data[each_column].to_list()
                for i in range(window_size - 1, 0, -1):
                    each_estimated = a_col[i]
                    if not np.isnan(each_estimated):
                        break
                estimated.append(each_estimated)
                distance += pow(original_data.iloc[0][each_column] - each_estimated, 2)
            
            distance = math.sqrt(distance) / len(columns2)  # type: ignore

            db.insert({'rnum': rnum, 'method': 'Sustain', 'window_size': window_size, 'columns': columns2, 'start_point': start_point, 'distance': None if distance is np.NaN else distance, 'original_data': original_data[columns].iloc[0].to_list(), 'estimated': estimated})

            del model, parser, samples, builder
        del current_data, original_data
    else:
        pass
    del Log, data