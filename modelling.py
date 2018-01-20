from fbprophet import Prophet


df = agg_db.rename(columns={'price_mean': 'y'})
df['ds'] = df.index

df = df[['ds', 'y']]
df = df.reset_index(drop=True)


#Create train / test
part_db = 0.2
l = len(df)
int_stop = int(l*(1 - 0.2))
db_train = df.iloc[0:int_stop,:]
db_test = df.iloc[int_stop:, :]



m = Prophet()

m.fit(db_train)

df.tail(5)
future = m.make_future_dataframe(periods=300, freq="2S")


forecast = m.predict(future)
forecast['yhat']



m.plot(forecast)
