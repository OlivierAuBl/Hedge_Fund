from fbprophet import Prophet
import matplotlib.pyplot as plt
from get_data import BuildDatabase
import numpy as np

symbol = 'BNBBTC'
c = BuildDatabase(api_key, api_secret, symbol)
db = c.start_data_extract(6000)

db = c.create_modeling_database('5s')




def build_target(database, t_interval, percentage_rise=3):
    price = database['price'].values
    new_target = np.zeros(len(price))

    for i, p in enumerate(price):
        price_normalized = price/p
        window_price = price_normalized[i:i + t_interval]

        if max(window_price) > (1. + percentage_rise/100.):
            new_target[i] = 1.

    return new_target

# Add target
ts = build_target(db, t_interval=30, percentage_rise=0.5)
ts.mean()
db['y'] = ts

#Create train / test
df = db.copy(deep=True)
part_db = 0.2
l = len(df)
int_stop = int(l*(1 - 0.2))
db_train = df.iloc[0:int_stop,:]
db_test = df.iloc[int_stop:, :]

########################################################################################################################
#                                      ------------------- Prophet -----------------

#   Problem since freq lower than day are not yet implemented. Should be in v. 0.3: check the releases
########################################################################################################################





########################################################################################################################
#                                      ------------------- Prophet -----------------

#   Problem since freq lower than day are not yet implemented. Should be in v. 0.3: check the releases
########################################################################################################################
df = db.rename(columns={'price': 'y'})
df['ds'] = df.index

df = df[['ds', 'y']]
df = df.reset_index(drop=True)
db_train.plot(x='ds', y="y")
db_test.plot(x='ds', y="y")


m = Prophet()

# m.add_seasonality(name='short', period=5, fourier_order=5)
m.add_seasonality(name='minute', period=1, fourier_order=5)
m.fit(db_train)

future = m.make_future_dataframe(periods=len(db_test), freq="2S")
forecast = m.predict(future)
m.plot(forecast)
m.plot_components(forecast)
#todo: learn on short and simulate a live trending + predicting

def simulate_live_modeling(df, len_train, len_pred, len_update):
    """each ten seconds, take the last 10 minutes, train the prophet and predict."""
    #len_train, len_pred, len_update = 10, 5, 4
    for i in range(10):
        #create live mini dbs
        train_range = df.iloc[i*len_update:i*len_update + len_train, :].ds
        window_train = [train_range.min(), train_range.max()]
        window_pred = [train_range.max(), ]
        train = df.loc[(df['ds'] >= window_train[0]) & (df['ds'] <= window_train[1]), :]

        pred_range = df.iloc[i*len_update + len_train:i*len_update + len_train+len_pred, :].ds
        window_pred = [pred_range.min(), pred_range.max()]
        pred = df.loc[(df['ds'] >= window_pred[0]) & (df['ds'] <= window_pred[1]), :]

        # create live models
        m = Prophet()
        m.fit(train)
        print train
        print pred

        future = m.make_future_dataframe(periods=len(pred), freq="2S")
        forecast = m.predict(future)

        plot_model(train, pred, forecast)


simulate_live_modeling(df, 400, 60, 200)



def plot_model(db_train, db_test, forecast):
    plt.plot(df.ds, df.y, 'ro', markersize=0.5)
    plt.plot(db_train['ds'], forecast.iloc[0:len(db_train), :]['yhat'], linewidth=2, color='C0')
    plt.plot(db_test['ds'], forecast.iloc[len(db_train):len(db_train)+len(db_test), :]['yhat'], linewidth=2, color='C1')
    plt.show()




plot_model(db_train, db_test, forecast)

forecast['yhat']



m.plot(forecast)
