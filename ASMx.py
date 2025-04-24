import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def add_bounded_edges(matrix, boundary_value, row_boundary_thickness, col_boundary_thickness):
    # function to add boundary to the matrix to avoid edge effect
    """
    @param matrix: A 2D numpy array to which the boundary will be added.
    @type matrix: numpy.array
    @param boundary_value: The value to fill the boundary with.
    @type boundary_value: float
    @param row_boundary_thickness: The thickness of the boundary to be added to the rows.
    @type row_boundary_thickness: int
    """
    original_rows, original_cols = matrix.shape
    new_rows = original_rows + 2 * row_boundary_thickness
    new_cols = original_cols + 2 * col_boundary_thickness

    # Create a new matrix filled with the boundary value
    new_matrix = np.full((new_rows, new_cols), boundary_value)

    # Insert the original matrix into the center of the new matrix
    new_matrix[row_boundary_thickness:row_boundary_thickness + original_rows,
               col_boundary_thickness:col_boundary_thickness + original_cols] = matrix

    return new_matrix

def generate_weight_matrices(smoothing_time_window, smoothing_space_window, delta=0.10, dx=0.02, dt=4, c_cong=12, c_free=-45, tau=9, plot=True):
    """
    Generate weight matrices for congestion and free flow conditions.
    @param delta: range of spatial smoothing in x.
    @type delta: float
    @param dx: The distance in miles between two adjacent grid points.
    @type dx: float
    @param dt: The time in seconds between two adjacent grid points.
    @type dt: int
    @param c_cong: The speed in miles per hour for the congestion condition.
    @type c_cong: int
    @param c_free: The speed in miles per hour for the free flow condition.
    @type c_free: int
    @param tau: The range of temporal smoothing in time, in seconds.
    @type tau: int
    @param plot: Whether to plot the weight matrices.
    @type plot: bool
    @return: The weight matrices for congestion and free flow conditions.
    @rtype: tuple of numpy.array
    Example:
        cong_weight_matrix, free_weight_matrix = generate_weight_matrices(delta=0.12, dx=0.02, dt=4, c_cong=13, c_free=-45, tau=20, plot=True)
    """
    t = smoothing_time_window
    x = smoothing_space_window
    x_mat = int(x / dx) * 2 + 1
    t_mat = int(t / dt) * 2 + 1
    matrix = np.zeros([x_mat, t_mat])
    matrix_df = pd.DataFrame(matrix)
    st_df = matrix_df.stack().reset_index()
    st_df.columns = ['x', 't', 'weight']
    st_df['time'] = dt * (st_df['t'] - int(t_mat / 2))
    st_df['space'] = dx * (st_df['x'] - int(x_mat / 2))

    def fill_cong_weight(row):
        t_new = row['time'] - row['space'] / (c_cong / 3600)
        return np.exp(-(abs(t_new) / tau + abs(row['space']) / delta))

    def fill_free_weight(row):
        t_new = row['time'] - row['space'] / (c_free / 3600)
        return np.exp(-(abs(t_new) / tau + abs(row['space']) / delta))

    st_df['cong_weight'] = st_df.apply(fill_cong_weight, axis=1)
    st_df['free_weight'] = st_df.apply(fill_free_weight, axis=1)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=20)
        plt.scatter(st_df.time, -st_df.space, c=st_df.cong_weight, vmax=1, vmin=0, s=50)
        plt.xlabel(r'\textbf{Time (seconds)}')
        plt.ylabel(r'\textbf{Space (mile)}')
        plt.title(r'\textbf{Congestion Weight}')
        plt.colorbar(label=r'\textbf{Weight}')
        plt.tight_layout()
        plt.savefig('congest.png', dpi=300, bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=20)
        plt.scatter(st_df.time, -st_df.space, c=st_df.free_weight, vmax=1, vmin=0, s=50)
        plt.xlabel(r'\textbf{Time (seconds)}')
        plt.ylabel(r'\textbf{Space (mile)}')
        plt.title(r'\textbf{Free Weight}')
        plt.colorbar(label=r'\textbf{Weight}')
        plt.tight_layout()
        plt.savefig('free.png', dpi=300, bbox_inches='tight')
        plt.show()

    cong_weight_matrix = st_df.pivot(index='t', columns='x', values='cong_weight').values
    free_weight_matrix = st_df.pivot(index='t', columns='x', values='free_weight').values

    return cong_weight_matrix, free_weight_matrix


from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm

def smooth_speed_field(raw_data, cong_weight_matrix, free_weight_matrix, vthr = 40, vdelta = 10):
    half_x_mat = int((cong_weight_matrix.shape[1] - 1) / 2)
    half_t_mat = int((cong_weight_matrix.shape[0] - 1) / 2)
    n_time, n_space = raw_data.shape

    raw_data_w_bound = add_bounded_edges(raw_data, np.nan, half_t_mat, half_x_mat)

    def process_time_idx(time_idx):
        row_result = np.zeros(n_space)
        for space_idx in range(n_space):
            neighbour_matrix = raw_data_w_bound[
                time_idx:time_idx + 2 * half_t_mat + 1,
                space_idx:space_idx + 2 * half_x_mat + 1
            ]
            neighbour_matrix = np.array(neighbour_matrix)
            mask = ~np.isnan(neighbour_matrix)
            neighbour_fillna = np.nan_to_num(neighbour_matrix, nan=0.0)

            N_cong = np.sum(mask * cong_weight_matrix)
            N_free = np.sum(mask * free_weight_matrix)

            if N_cong == 0:
                v_cong = np.nan
            else:
                v_cong = np.sum(neighbour_fillna * cong_weight_matrix) / N_cong

            if N_free == 0:
                v_free = np.nan
            else:
                v_free = np.sum(neighbour_fillna * free_weight_matrix) / N_free

            if N_cong != 0 and N_free != 0:
                w = 0.5 * (1 + np.tanh((vthr - min(v_cong, v_free)) / vdelta))
                v = w * v_cong + (1 - w) * v_free
            elif N_cong == 0:
                v = v_free
            elif N_free == 0:
                v = v_cong
            else:
                v = np.nan

            row_result[space_idx] = v
        return row_result

    result_rows = Parallel(n_jobs=-1)(
        delayed(process_time_idx)(time_idx) for time_idx in tqdm(range(n_time))
    )
    smooth_data = np.array(result_rows)
    return smooth_data



def matrix_to_coordinates(matrix):
    """
    Converts a 2D matrix (numpy.array) into a list of coordinates with values.

    This function iterates through each element of a 2D matrix and
    creates a list of coordinates, where each coordinate is represented
    as a list containing the row index, column index, and the value at
    that position in the matrix.

    :param matrix: A numpy.array where each sublist represents a row in the matrix.
    :type matrix: numpy.array filled with float
    :return: A list of coordinates, where each coordinate is a list of [row_index, column_index, value].
    :rtype: list of list of float

    Example:
        matrix = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        coords = matrix_to_coordinates(matrix) print(coords)  # Output: [[0, 0, 1], [0, 1, 2], [0, 2, 3], [1, 0, 4],
        [1, 1, 5], [1, 2, 6], [2, 0, 7], [2, 1, 8], [2, 2, 9]]
    """
    coordinates = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            coordinates.append([i, j, matrix[i][j]])
    return coordinates

def asm_data_w_x(processed_data, smoothing_time_window, smoothing_space_window, delta, tau, c_cong=12, c_free=-45, dx=0.02, dt=4, data_columns=['speed', 'occ', 'volume']):
    t = smoothing_time_window
    x = smoothing_space_window
    x_mat = int(x / dx) * 2 + 1
    t_mat = int(t / dt) * 2 + 1
    matrix = np.zeros([x_mat, t_mat])
    matrix_df = pd.DataFrame(matrix)
    st_df = matrix_df.stack().reset_index()
    st_df.columns = ['x', 't', 'weight']
    st_df['time'] = dt * (st_df['t'] - int(t_mat / 2))
    st_df['space'] = dx * (st_df['x'] - int(x_mat / 2))

    def fill_cong_weight(row):
        t_new = row['time'] - row['space'] / (c_cong / 3600)
        return np.exp(-(abs(t_new) / tau + abs(row['space']) / delta))

    def fill_free_weight(row):
        t_new = row['time'] - row['space'] / (c_free / 3600)
        return np.exp(-(abs(t_new) / tau + abs(row['space']) / delta))

    st_df['cong_weight'] = st_df.apply(fill_cong_weight, axis=1)
    st_df['free_weight'] = st_df.apply(fill_free_weight, axis=1)
    cong_weight_matrix = st_df.pivot(index='t', columns='x', values='cong_weight').values
    free_weight_matrix = st_df.pivot(index='t', columns='x', values='free_weight').values
    half_x_mat = int((cong_weight_matrix.shape[1] - 1) / 2)
    half_t_mat = int((cong_weight_matrix.shape[0] - 1) / 2)
    # Assuming only lane 1 is used here
    lanes = [1]
    data_columns = data_columns

    data = processed_data[
        ['milemarker', 'time_unix_fix'] + [f'lane{lane}_{col}' for lane in lanes for col in data_columns]]

    min_milemarker = 58.7
    max_milemarker = 62.7
    min_time_unix = data['time_unix_fix'].min()
    max_time_unix = data['time_unix_fix'].max()

    milemarkers = np.arange(min_milemarker, max_milemarker, dx)
    time_range_unix = np.arange(min_time_unix, max_time_unix, dt)
    space_time_matrix_unix = pd.DataFrame(index=time_range_unix, columns=milemarkers)

    def fill_space_time_matrix(data, lane, data_type):
        matrix = space_time_matrix_unix.copy()
        for index, row in data.iterrows():
            time_index = row['time_unix_fix']
            milemarker_index = row['milemarker']
            nearest_time = matrix.index.get_indexer([time_index], method='nearest')[0]
            nearest_milemarker = matrix.columns.get_indexer([milemarker_index], method='nearest')[0]
            matrix.iloc[nearest_time, nearest_milemarker] = row[f'lane{lane}_{data_type}']
        return matrix

    smoothed_data = {}
    for lane in lanes:
        for data_type in data_columns:
            print(f'Processing lane {lane} {data_type}...')
            if data_type == 'speed':
                space_time_matrix = fill_space_time_matrix(data, lane, data_type)
                pre_smoothed_data = pd.DataFrame(space_time_matrix.values)
                pre_smoothed_data_w_bound = add_bounded_edges(pre_smoothed_data, np.nan, half_t_mat, half_x_mat)
                n_time = pre_smoothed_data.shape[0]
                n_space = pre_smoothed_data.shape[1]

                # Parallelize over time indices.
                def process_time_idx(time_idx):
                    row_result = np.zeros(n_space)
                    for space_idx in range(n_space):
                        neighbour_matrix = pre_smoothed_data_w_bound[
                            time_idx:time_idx + 2 * half_t_mat + 1,
                            space_idx:space_idx + 2 * half_x_mat + 1]
                        neighbour_matrix = np.array(neighbour_matrix)
                        mask = ~np.isnan(neighbour_matrix)
                        neighbour_fillna = np.nan_to_num(neighbour_matrix, nan=0.0)
                        N_cong = np.sum(mask * cong_weight_matrix)
                        N_free = np.sum(mask * free_weight_matrix)
                        if N_cong == 0:
                            v_cong = np.nan
                        else:
                            v_cong = np.sum(neighbour_fillna * cong_weight_matrix) / N_cong
                        if N_free == 0:
                            v_free = np.nan
                        else:
                            v_free = np.sum(neighbour_fillna * free_weight_matrix) / N_free
                        if N_cong != 0 and N_free != 0:
                            w = 0.5 * (1 + np.tanh((40 - min(v_cong, v_free)) / 10))
                            v = w * v_cong + (1 - w) * v_free
                        elif N_cong == 0:
                            v = v_free
                        elif N_free == 0:
                            v = v_cong
                        else:
                            v = np.nan
                        row_result[space_idx] = v
                    return row_result

                result_rows = Parallel(n_jobs=-1)(delayed(process_time_idx)(time_idx) for time_idx in range(n_time))
                smooth_data = np.array(result_rows)
                smoothed_data[(lane, data_type)] = smooth_data

            else:  # For data types other than 'speed'
                space_time_matrix = fill_space_time_matrix(data, lane, data_type)
                pre_smoothed_data = pd.DataFrame(space_time_matrix.values)
                pre_smoothed_data_w_bound = add_bounded_edges(pre_smoothed_data, np.nan, half_t_mat, half_x_mat)
                
                # Reuse the already computed smoothed speed data for weighting.
                pre_smoothed_speed = pd.DataFrame(smoothed_data[(lane, 'speed')])
                pre_smoothed_data_speed_w_bound = add_bounded_edges(pre_smoothed_speed, np.nan, half_t_mat, half_x_mat)
                
                n_time = pre_smoothed_data.shape[0]
                n_space = pre_smoothed_data.shape[1]
                
                def process_time_idx_non_speed(time_idx):
                    row_result = np.zeros(n_space)
                    for space_idx in range(n_space):
                        neighbour_matrix = pre_smoothed_data_w_bound[
                            time_idx:time_idx + 2 * half_t_mat + 1,
                            space_idx:space_idx + 2 * half_x_mat + 1]
                        neigbour_matrix_speed = pre_smoothed_data_speed_w_bound[
                            time_idx:time_idx + 2 * half_t_mat + 1,
                            space_idx:space_idx + 2 * half_x_mat + 1]
                        neighbour_matrix = np.array(neighbour_matrix)
                        mask = ~np.isnan(neighbour_matrix)
                        neighbour_fillna = np.nan_to_num(neighbour_matrix, nan=0.0)
                        
                        neigbour_matrix_speed = np.array(neigbour_matrix_speed)
                        mask_speed = ~np.isnan(neigbour_matrix_speed)
                        neighbour_fillna_speed = np.nan_to_num(neigbour_matrix_speed, nan=0.0)
                        
                        N_cong = np.sum(mask * cong_weight_matrix)
                        N_free = np.sum(mask * free_weight_matrix)
                        N_cong_speed = np.sum(mask_speed * cong_weight_matrix)
                        N_free_speed = np.sum(mask_speed * free_weight_matrix)
                        if N_cong == 0:
                            v_cong = np.nan
                        else:
                            v_cong = np.sum(neighbour_fillna * cong_weight_matrix) / N_cong
                            v_cong_speed = np.sum(neighbour_fillna_speed * cong_weight_matrix) / N_cong_speed
                        if N_free == 0:
                            v_free = np.nan
                        else:
                            v_free = np.sum(neighbour_fillna * free_weight_matrix) / N_free
                            v_free_speed = np.sum(neighbour_fillna_speed * free_weight_matrix) / N_free_speed
                        if N_cong != 0 and N_free != 0:
                            w = 0.5 * (1 + np.tanh((40 - min(v_cong_speed, v_free_speed)) / 10))
                            v = w * v_cong + (1 - w) * v_free
                        elif N_cong == 0:
                            v = v_free
                        elif N_free == 0:
                            v = v_cong
                        else:
                            v = np.nan
                        row_result[space_idx] = v
                    return row_result

                result_rows = Parallel(n_jobs=-1)(delayed(process_time_idx_non_speed)(time_idx) for time_idx in range(n_time))
                smooth_data = np.array(result_rows)
                smoothed_data[(lane, data_type)] = smooth_data

    # # If there are still NaNs in the smoothed data, fill them with the nearest values.
    # smoothed_data = {
    #     key: pd.DataFrame(value).fillna(method='ffill').fillna(method='bfill').values 
    #     for key, value in smoothed_data.items()
    # }
    # Convert the smoothed data to coordinates.
    result_dfs = []
    for lane in lanes:
        for data_type in data_columns:
            smooth_data = smoothed_data[(lane, data_type)]
            smooth_data_df = pd.DataFrame(matrix_to_coordinates(smooth_data))
            smooth_data_df.columns = ['time_index', 'space_index', f'lane{lane}_{data_type}']
            smooth_data_df['unix_time'] = min_time_unix + smooth_data_df['time_index'] * dt + dt / 2
            smooth_data_df['milemarker'] = min_milemarker + smooth_data_df['space_index'] * dx + dx / 2
            result_dfs.append(smooth_data_df[['unix_time', 'milemarker', f'lane{lane}_{data_type}']])
    # Concatenate all results into a single DataFrame.
    final_df = pd.concat(result_dfs, axis=1)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    return final_df
