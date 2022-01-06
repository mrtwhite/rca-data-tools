#!/usr/local/bin/python3

# -*- coding: utf-8 -*-



# import packages

import dask
import datetime
import gc
import math
import numba
import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from functools import reduce



@numba.njit
def execute_decimation(threshold, every, data, sampled, time_array, a):
    for i in range(0, threshold - 2):
        # Calculate point average for next bucket (containing c)
        avg_x = 0
        avg_y = 0
        avg_range_start = int(math.floor((i + 1) * every) + 1)
        avg_range_end = int(math.floor((i + 2) * every) + 1)
        avg_rang_end = avg_range_end if avg_range_end < len(data) else len(data)

        avg_range_length = avg_rang_end - avg_range_start

        while avg_range_start < avg_rang_end:
            avg_x += data[avg_range_start][0]
            avg_y += data[avg_range_start][1]
            avg_range_start += 1

        avg_x /= avg_range_length
        avg_y /= avg_range_length

        # Get the range for this bucket
        range_offs = int(math.floor((i + 0) * every) + 1)
        range_to = int(math.floor((i + 1) * every) + 1)

        # Get middle of bucket
        bucket_middle = math.floor(
            np.median(np.array([j for j in range(range_offs, range_to + 1)]))
        )

        time_array.append(data[bucket_middle][0])
        # Point a
        point_ax = data[a][0]
        point_ay = data[a][1]

        max_area = -1

        while range_offs < range_to:
            # Calculate triangle area over three buckets
            area = (
                math.fabs(
                    (point_ax - avg_x) * (data[range_offs][1] - point_ay)
                    - (point_ax - data[range_offs][0]) * (avg_y - point_ay)
                )
                * 0.5
            )

            if area > max_area:
                max_area = area
                max_area_point = data[range_offs]
                next_a = range_offs  # Next a is this b
            range_offs += 1
        sampled.append(max_area_point[1])  # Pick this point from the bucket
        a = next_a  # This a is the next a (chosen b)

    return sampled, time_array


class LttbException(Exception):
    pass


def largest_triangle_three_buckets(data, threshold):
    """
    Return a downsampled version of data.
    Original code found at https://github.com/devoxi/lttb-py.
    
    Args:
        data: Original data that will be decimated.
              Must be a numpy array or list of lists.
              Data must be formatted this way: [[x,y], [x,y], [x,y], ...]
        threshold (int): threshold must be >= 2 and <= to the len of data.
    Returns:
        numpy.array: Decimated data.
    """

    # --- Initial checks ---
    #     if not isinstance(data, (list, np.ndarray)):
    #         raise LttbException("data is not a list or numpy array")
    #     if not isinstance(threshold, int) or threshold <= 2 or threshold >= len(data):
    #         raise LttbException("threshold not well defined")
    #     for i in data:
    #         if not isinstance(i, (list, np.ndarray)) or len(i) != 2:
    #             raise LttbException("datapoints are not lists or numpy array")

    try:
        data = data.compute()
        # Bucket size. Leave room for start and end data points
        every = (len(data) - 2) / (threshold - 2)

        a = 0  # Initially a is the first point in the triangle

        # Always add the first point
        sampled = [data[0][1]]
        time_array = [data[0][0]]

        # --- Perform Decimation ---
        sampled, time_array = execute_decimation(
            threshold, every, data, sampled, time_array, a
        )

        # Always add the last point
        sampled.append(data[len(data) - 1][1])
        time_array.append(data[len(data) - 1][0])

        decimated_data = np.array(list(zip(time_array, sampled)))

        return decimated_data
    except Exception as e:
        raise LttbException(e)


def perform_decimation(ds, threshold):
    time_da = ds.time.astype(int)
    cols = [time_da.name, ds.name]
    da_data = dask.array.stack([time_da.data, ds.data], axis=1)
    decdata = largest_triangle_three_buckets(da_data, threshold)
    #client.cancel(da_data)
    del ds
    gc.collect()
    return pd.DataFrame(decdata, columns=cols)

def downsample(raw_ds, threshold):
    print(f"{datetime.datetime.now().strftime('%H:%M:%S')}:    Get list of data arrays")
    da_list = (raw_ds[var] for var in raw_ds)

    df_list = []
    for da in da_list:
        print(
            f"{datetime.datetime.now().strftime('%H:%M:%S')}:    Executing decimation for {da.name}"
        )
        decdf = perform_decimation(da, threshold)
        df_list.append(decdf)

        del decdf
        gc.collect()
    print(
        f"{datetime.datetime.now().strftime('%H:%M:%S')}:    Decimation process completed."
    )

    print(
        f"{datetime.datetime.now().strftime('%H:%M:%S')}:    Creating decimated dataframe."
    )
    final_df = reduce(lambda left, right: pd.merge(left, right, on="time"), df_list)
    final_df['time'] = pd.to_datetime(final_df['time'])

    return final_df

