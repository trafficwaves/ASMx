import pandas as pd
import numpy as np
import pytz
import datetime
import ast
from datetime import timedelta
from functools import reduce
import matplotlib.pyplot as plt

def preprocess_raw_data(file_path):
    import datetime
    import pytz
    def time_trans(dt_str):
        # Define the date-time string
        # Convert the string to a datetime object in CDT
        dt = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        #     cdt = pytz.timezone('America/Chicago')
        #     dt = cdt.localize(dt)
        # Convert to UTC and get the Unix timestamp
        unix_timestamp = dt.astimezone(pytz.UTC).timestamp()
        return unix_timestamp
    data = pd.read_csv(file_path)
    return data


def time_trans(dt_str):
    # Define the date-time string
    # Convert the string to a datetime object in CDT
    dt = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    #     cdt = pytz.timezone('America/Chicago')
    #     dt = cdt.localize(dt)
    # Convert to UTC and get the Unix timestamp
    unix_timestamp = dt.astimezone(pytz.UTC).timestamp()
    return unix_timestamp


def round_to_fix(time):
    seconds = time.second
    microseconds = time.microsecond
    total_seconds = seconds + microseconds / 1_000_000.0

    if total_seconds % 60 < 15:
        new_second = 0
    elif 15 <= total_seconds % 60 < 45:
        new_second = 30
    else:
        new_second = 0
        time = time + timedelta(minutes=1)

    return time.replace(second=new_second, microsecond=0).timestamp()


def extract_lane_speed(lane_dict_str, lane_num):
    # Convert the string representation of dictionary to actual dictionary
    lane_dict = ast.literal_eval(lane_dict_str)

    # Search for the lane and extract the speed
    for value in lane_dict.values():
        if f"Lane{lane_num}" in value[0] and len(value) > 1:
            return value[1]
    return None


def extract_lane_volume(lane_dict_str, lane_num):
    # Convert the string representation of dictionary to actual dictionary
    lane_dict = ast.literal_eval(lane_dict_str)

    # Search for the lane and extract the speed
    for value in lane_dict.values():
        if f"Lane{lane_num}" in value[0] and len(value) > 1:
            return value[2]
    return None

def extract_lane_occupancy(lane_dict_str, lane_num):
    # Convert the string representation of dictionary to actual dictionary
    lane_dict = ast.literal_eval(lane_dict_str)

    # Search for the lane and extract the speed
    for value in lane_dict.values():
        if f"Lane{lane_num}" in value[0] and len(value) > 1:
            return value[3]
    return None

def read_data_fix_time(file_path):
    data = pd.read_csv(file_path)
    data['time_unix'] = data['link_update_time'].apply(lambda x: time_trans(x[:19]))
    data['time_unix_fix'] = data['time_unix'].apply(lambda x: round_to_fix(datetime.datetime.fromtimestamp(x)))
    return data

def raw_lane_level_df_process(data):
    def extract_lane_feature(lane_dict_str, lane_num, feature_index):
        lane_dict = ast.literal_eval(lane_dict_str)
        for value in lane_dict.values():
            if f"Lane{lane_num}" in value[0] and len(value) > feature_index:
                return value[feature_index]
        return None
    def extract_lane_df(data, lane_number):
        lane = data[data['lane_dict_text'].str.contains(f"Lane{lane_number}")].copy()
        lane[f'lane{lane_number}_speed'] = lane['lane_dict_text'].apply(lambda x: extract_lane_feature(x, lane_number, 1))
        lane[f'lane{lane_number}_volume'] = lane['lane_dict_text'].apply(lambda x: extract_lane_feature(x, lane_number, 2))
        lane[f'lane{lane_number}_occ'] = lane['lane_dict_text'].apply(lambda x: extract_lane_feature(x, lane_number, 3))
        return lane[['time_unix_fix', 'milemarker', f'lane{lane_number}_speed', f'lane{lane_number}_volume', f'lane{lane_number}_occ']]
    # Function to merge multiple DataFrames
    def merge_dataframes(dfs, join_keys, how='outer'):
        return reduce(lambda left, right: pd.merge(left, right, on=join_keys, how=how), dfs)
    # List of DataFrames to merge
    lane_dataframes = [extract_lane_df(data, i) for i in range(1, 5)]
    # Keys to merge on
    join_keys = ['time_unix_fix', 'milemarker']
    # Merging all lanes
    all_lanes = merge_dataframes(lane_dataframes, join_keys)
    # to create another with milemaker and time_unix_fix[:-1]
    unique_milemarkers = all_lanes['milemarker'].unique()
    # Extract unique time_unix_fix, excluding the last one
    unique_time_unix_fix = np.sort(all_lanes['time_unix_fix'].unique())[:-1]
    # Create a DataFrame for each unique value
    milemarker_df = pd.DataFrame({'milemarker': unique_milemarkers})
    time_unix_fix_df = pd.DataFrame({'time_unix_fix': unique_time_unix_fix})
    # Perform a cross join to get all combinations
    all_lanes_clean = milemarker_df.merge(time_unix_fix_df, how='cross')
    all_lanes_unique = all_lanes.drop_duplicates(subset=['milemarker', 'time_unix_fix'])
    merged = pd.merge(all_lanes_clean, all_lanes_unique, on=['milemarker', 'time_unix_fix'], how='left')
    for i in range(1, 5):
        speed_col = f'lane{i}_speed'
        volume_col = f'lane{i}_volume'
        occ_col = f'lane{i}_occ'
        merged.loc[merged[speed_col].isna(), [volume_col, occ_col]] = np.nan
    for feature in ['speed', 'volume', 'occ']:
        columns = [f'lane{i}_{feature}' for i in range(1, 5)]
        # Calculate the mean of all lanes for the current feature, ignoring NaNs
        overall_mean = merged[columns].mean(axis=1, skipna=True)
        for column in columns:
            merged[column].fillna(overall_mean, inplace=True)
    return merged