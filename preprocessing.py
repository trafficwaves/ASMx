import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def fill_space_time_matrix(raw_data, dx = 0.1, dt = 10, lane = 1, data_type = 'speed'):
    data = raw_data[['milemarker', 'time_unix_fix', f'lane{lane}_{data_type}']].copy()
    min_milemarker = 58.7
    max_milemarker = 62.7
    min_time_unix = int(data['time_unix_fix'].min())
    max_time_unix = int(data['time_unix_fix'].max()) + 30
    milemarkers = np.arange(min_milemarker, max_milemarker, dx)
    # print(milemarkers)
    time_range_unix = np.arange(min_time_unix, max_time_unix, dt)
    matrix = pd.DataFrame(index=time_range_unix, columns=milemarkers)
    for index, row in data.iterrows():
        time_index = row['time_unix_fix']
        milemarker_index = row['milemarker']
        nearest_time = matrix.index.get_indexer([time_index], method='nearest')[0]
        nearest_milemarker = matrix.columns.get_indexer([milemarker_index], method='nearest')[0]
        matrix.iloc[nearest_time, nearest_milemarker] = row[f'lane{lane}_{data_type}']
    return matrix


val = pd.read_csv('dates.csv')
dates = val.date.tolist()
for lane in range(1, 5):
    for date in dates:
        print('date:', date)
        raw_data = pd.read_csv(f'data/raw_data/rds/{date}.csv', low_memory=False)
        speed = fill_space_time_matrix(raw_data, dx = 0.02, dt = 4, lane = 1, data_type = 'speed')
        output = speed.T.values.copy()
        output = output.astype(np.float32)
        print(f'lane {lane}, {date} output shape: {output.shape}')
        output_dir = f'data/processed_data/rds/lane{lane}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(f'{output_dir}/{date}.npy', output)
        # num of nan
        # print(f'lane {lane}, {date} num of nan: {np.isnan(output).sum()}')
        plt.imshow(output, cmap='hot', interpolation='nearest',aspect='auto')
        plt.colorbar()
        plt.title(f'Lane {lane} Speed Matrix on {date}')
        plt.xlabel('Space')
        plt.ylabel('Time')
        plt.savefig(f'demo_raw_{lane}_{date}.png')
        plt.close()
