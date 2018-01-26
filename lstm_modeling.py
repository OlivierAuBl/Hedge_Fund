

########################################################################################################################
#                                      ------------------- LSTM -----------------

########################################################################################################################
# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

def add_previous_values(df, target_name, lag=1):

    new_col = {}
    for i in range(lag):
        new_col['{}-{}'.format(target_name, i)] = df[target_name].shift(i)
    new_col = pd.DataFrame(new_col)
    df = pd.concat([df, new_col.fillna(0)], axis=1) #todo: fillna not good
    return df

test = add_previous_values(db_train, 'price', lag=3)

#todo: stationnarize ?

#todo: what has to be btw -1 and 1 ??

#check keras doc...

X, y = db_train[:, 0:-1], db_train[:, -1]
X = X.reshape(X.shape[0], 1, X.shape[1])


