# Convert the .csv files to .feather

if __name__ == '__main__':

    import pandas
    import pyarrow

    train = pd.read_csv('/data/train.csv')
    test = pd.read_csv('/data/test.csv')

    # Save Memory
    for col, dtype in train.dtypes.iteritems():
        if dtype.name.startswith('int'):
            train[col] = pd.to_numeric(train[col], downcast ='integer')
        elif dtype.name.startswith('float'):
            train[col] = pd.to_numeric(train[col], downcast ='float')

    for col, dtype in test.dtypes.iteritems():
        if dtype.name.startswith('int'):
            test[col] = pd.to_numeric(test[col], downcast ='integer')
        elif dtype.name.startswith('float'):
            test[col] = pd.to_numeric(test[col], downcast ='float')

    train.to_feather('/data/train.feather')
    test.to_feather('/data/test.feather')
