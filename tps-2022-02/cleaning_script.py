import numpy as np
import pandas as pd
import gc
import time

# Load original .csv data
start = time.time()
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Function for downcasting float/int datatypes
def reduce_memory_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col, dtype in df.dtypes.iteritems():
        if dtype.name.startswith('int'):
            df[col] = pd.to_numeric(df[col], downcast ='integer')
        elif dtype.name.startswith('float'):
            df[col] = pd.to_numeric(df[col], downcast ='float')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

# Reduce memory and save as .feather
train = reduce_memory_usage(train)
test = reduce_memory_usage(test)
train.to_feather('data/train.feather')
test.to_feather('data/test.feather')
end = time.time()
print(f'Done in {round(end-start,2)}s.')
