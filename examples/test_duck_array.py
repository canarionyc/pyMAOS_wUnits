import numpy as np

# These are common duck arrays
import pandas as pd
import dask.array as da
import xarray as xr

# Create a NumPy array
numpy_array = np.array([1, 2, 3])

# Create a pandas Series (duck array)
pandas_series = pd.Series([1, 2, 3])

# Create a Dask array (duck array)
dask_array = da.array([1, 2, 3])

# Example function that works with both NumPy arrays and duck arrays
def process_array(arr):
    # This will work with NumPy arrays and duck arrays
    # but would fail with regular Python lists
    return arr + 1

# These all work because they implement array-like interfaces
result_numpy = process_array(numpy_array)
result_pandas = process_array(pandas_series)
result_dask = process_array(dask_array)

print(f"NumPy result: {result_numpy}")
print(f"Pandas result: {result_pandas}")
print(f"Dask result: {result_dask}")  # Note: Dask is lazy, needs .compute()