import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Pool
import preprocessing.FAST as FAST
import os
import warnings
warnings.filterwarnings("ignore")

# Define the tdot_milemarker and closest_mm lists
tdot_milemarker = [
    53, 53.3, 53.6, 53.9, 54.1, 54.6, 55, 55.3, 55.5, 56, 56.3, 56.7, 57.3, 57.7,
    58.1, 58.3, 58.6, 58.8, 59, 59.3, 59.7, 60, 60.4, 60.5, 61, 61.5, 62.1, 62.5,
    63.1, 63.6, 64, 64.3, 64.5, 64.8, 65, 65.1, 65.6, 65.9, 66.3, 66.5, 66.7,
    66.9, 67.3, 67.8, 68.2, 68.5, 68.8, 69.3, 69.8, 70.1
]

closest_mm = [
    53, 53.3, 53.61, 53.9, 54.16, 54.65, 55.02, 55.26, 55.59, 56.01, 56.52, 56.99,
    57.36, 57.68, 58.08, 58.26, 58.63, 58.8, 59.14, 59.28, 59.73, 59.97, 60.37,
    60.55, 61, 61.6, 62.23, 62.63, 63.08, 63.55, 64.01, 64.27, 64.47, 64.74,
    65.01, 65.13, 65.6, 65.93, 66.25, 66.47, 66.73, 66.95, 67.33, 67.8, 68.19,
    68.5, 68.9, 69.27, 69.78, 70.09
]
mm = pd.DataFrame({
    "tdot_milemarker": tdot_milemarker,
    "closest_mm": closest_mm
})

# write a function to transform the YYYY-MM-DD HH:MM:SS in str to unix time in Central Chicago Time
def time_to_unix(time_str):
    import time
    import pytz
    from datetime import datetime
    from pytz import timezone
    # define the timezone
    tz = timezone('America/Chicago')
    # define the format of the time
    fmt = '%Y-%m-%d %H:%M:%S'
    # define the time in the timezone
    time = tz.localize(datetime.strptime(time_str, fmt))
    # return the unix time
    return time.timestamp() + 20
# split the date from entry
def split_date(entry):
    return entry.split('.')[0]

rds_data_path = 'data/raw_record/rds/'

# # Check the files in the rds_data_path
# entries = os.listdir(rds_data_path)
# if '.DS_Store' in entries:
#     entries.remove('.DS_Store')
# entries.sort()

# Define a function to process each entry
def process_entry(entry):
    date = split_date(entry)
    print(f'processing the data on {date}')
    # read the data
    data = FAST.read_data_fix_time(f'{rds_data_path}{entry}.csv')
    processed_data = FAST.raw_lane_level_df_process(data)
    # Create a mapping from tdot_milemarker to closest_mm
    mapping = dict(zip(tdot_milemarker, closest_mm))
    # rename the milemarker column to 'tdot_milemarker'
    processed_data = processed_data.rename(columns={'milemarker': 'tdot_milemarker'})
    # Add a new column 'milemarker' in processed_data using the mapping
    processed_data['milemarker'] = processed_data['tdot_milemarker'].map(mapping)
    # make the value less than 0 to be NaN
    processed_data.loc[processed_data.lane1_speed < 0, 'lane1_speed'] = np.nan
    processed_data.loc[processed_data.lane1_volume < 0, 'lane1_volume'] = np.nan
    processed_data.loc[processed_data.lane1_occ < 0, 'lane1_occ'] = np.nan
    processed_data = processed_data[(processed_data.milemarker >= 53.3) & (processed_data.milemarker <= 70.1)].reset_index(drop=True)
    # make the time larger than 6:00 less than 10:00
    # get the unix time of that day's 6:00 and 10:00
    start_time = time_to_unix(f'{date} 06:00:00')
    end_time = time_to_unix(f'{date} 10:00:00')
    # convert the time to unix time
    processed_data = processed_data[(processed_data.time_unix_fix >= start_time) & (processed_data.time_unix_fix <= end_time)].reset_index(drop=True)
    if not os.path.exists(f'data/corridor_data/rds'):
        os.makedirs(f'data/corridor_data/rds')
    processed_data.to_csv(f'data/corridor_data/rds/{date}.csv', index=False)


# Use multiprocessing to process the entries concurrently
if __name__ == '__main__':
    val = pd.read_csv('dates.csv')
    val_entries = val.date.tolist()
    # with Pool(multiprocessing.cpu_count()) as pool:
    #     pool.map(process_entry, val_entries)
    for entry in val_entries:
        process_entry(entry)