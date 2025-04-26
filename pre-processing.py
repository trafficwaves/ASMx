import numpy as np
import pandas as pd


def fill_space_time_matrix(raw_data, dx = 0.1, dt = 10, lane = 1, data_type = 'speed'):
    data = raw_data[['milemarker', 'time_unix_fix', f'lane{lane}_{data_type}']].copy()
    min_milemarker = round(data['milemarker'].min(),0)
    max_milemarker = round(data['milemarker'].max(),0)
    min_time_unix = int(data['time_unix_fix'].min())
    max_time_unix = int(data['time_unix_fix'].max()) + 30
    milemarkers = np.round(np.arange(min_milemarker, max_milemarker, dx), 2)
    time_range_unix = np.round(np.arange(min_time_unix, max_time_unix, dt), 2)
    matrix = pd.DataFrame(index=time_range_unix, columns=milemarkers)
    for index, row in data.iterrows():
        time_index = row['time_unix_fix']
        milemarker_index = row['milemarker']
        nearest_time = matrix.index.get_indexer([time_index], method='nearest')[0]
        nearest_milemarker = matrix.columns.get_indexer([milemarker_index], method='nearest')[0]
        matrix.iloc[nearest_time, nearest_milemarker] = row[f'lane{lane}_{data_type}']
    return matrix

raw_data = pd.read_csv('data/2023-10-26.csv',low_memory=False)
speed = fill_space_time_matrix(raw_data, dx = 0.1, dt = 30, lane = 1, data_type = 'speed')
output = speed.T.values.copy()
# make output to be float32
output = output.astype(np.float32)
np.save('data/speed.npy', output)