from binance.client import Client
import pandas as pd
import matplotlib.pyplot as plt

class BuildDatabase:

    def __init__(self, api_key, api_secret, symbol):
        self.client = Client(api_key, api_secret)
        self.symbol = symbol
        self.database = pd.DataFrame(columns=['id', 'isBestMatch', 'isBuyerMaker', 'price', 'qty', 'time'])

    def start_data_extract(self, nrows):
        """
        Function used to create the raw database from the API
        :param nrows: int
            Number of rows requested in the database. It needs to be a multiple of 500.
        :return:
            return the database extracted from the API
        """
        # Get the ID now
        last_id = self.client.get_recent_trades(symbol=self.symbol, limit=1)[0]['id']
        start_id = last_id - nrows + 1

        #TODO: add condition not to scrape more than the API limit !!!!!!! (1200 to check)
        if (nrows % 500 == 0) or (nrows<1200*500):
            print("Data requested to the API from Id = {} to Id = {}".format(start_id, last_id))
        else:
            raise AttributeError('Fuck you, you have to put a multiple of 500 for nrows, '
                                 'or requested more than 1200$500 rows, read the doc next time.')

        for i in range(start_id, last_id, 500):  # 500 is max limit
            extract_i = self.client.get_historical_trades(symbol=self.symbol, limit=500, fromId=i)
            self.database = pd.concat([self.database, pd.DataFrame(extract_i)])

        return self.database

    def create_modeling_database(self, agg_lvl='10s'):
        """

        :param agg_lvl: str
        time use for aggregation
        :return:
        the aggregatd database with new features built.
        """
        db = self.database
        db["date"] = pd.to_datetime(db['time'], unit='ms')
        db.index = db["date"]
        db = db.drop(["date", "time", "id"], axis=1)

        db = db.apply(pd.to_numeric)  # todo: put float 32 + comment

        # create not bool columns
        db = db.rename(columns={'isBestMatch': 'isBestMatch_T', 'isBuyerMaker': 'isBuyerMaker_T'})
        db['isBestMatch_F'] = not (db['isBestMatch_T']).all()
        db['isBuyerMaker_F'] = not (db['isBuyerMaker_T']).all()

        # Create new columns
        # Real exchanged quantity: price_x_qty
        db['value_exchanged'] = db['price'] * db['qty']
        # Quanty sold / bought
        db["qty_sold"] = db['qty'] * db['isBuyerMaker_T']
        db["qty_bought"] = db['qty'] * db['isBuyerMaker_F']
        # Value sold / bought
        db['value_sold'] = db['value_exchanged'] * db['isBuyerMaker_T']
        db['value_bought'] = db['value_exchanged'] * db['isBuyerMaker_F']
        # Price variation / second derivative
        db['price_variation'] = db["price"].diff()
        db['price_variation_diff'] = db["price"].diff().diff()

        # Threshold;
        thr_price_variation = db['price_variation'].quantile(0.9)
        db['price_variation_thr'] = db['price_variation'] > thr_price_variation

        # aggregation columns:
        t_agg = agg_lvl
        agg_db = db.resample(t_agg).sum().interpolate(method='zero')

        for col in db.columns:

            if db[col].dtypes != 'bool':
                agg_db[col + '_max'] = db[col].resample(t_agg).max().interpolate(method='zero') #todo: is this interpo the best ?
                agg_db[col + '_min'] = db[col].resample(t_agg).min().interpolate(method='zero')
                agg_db[col + '_mean'] = db[col].resample(t_agg).mean().interpolate(method='zero')

        # price not summed and need to interpolate if no change in the interval
        agg_db['price'] = agg_db['price_mean']
        agg_db.drop('price_mean', axis=1, inplace=True)
        agg_db['price'] = agg_db['price'].interpolate(method='zero')


        # Trend last x seconds
        var = 'price_variation'
        for t in [10, 20, 40, 60, 120]:
            len_time = '{}S'.format(t)  # T for minutes / s for seconds
            mini_db_agg = pd.DataFrame(db[var].resample(len_time).sum())
            mini_db_agg.rename(columns={var: '{}_trend_{}'.format(var, len_time)}, inplace=True)
            mini_db_agg = mini_db_agg.resample(t_agg).max()
            mini_db_agg = mini_db_agg.interpolate(method='zero')
            agg_db = pd.merge(agg_db, mini_db_agg, how='left', left_index=True, right_index=True)

        agg_db["weekday_name"] = agg_db.index
        agg_db["weekday_name"] = agg_db["weekday_name"].dt.weekday_name
        # todo : reflechir aux NaN à la fin.
        # todo : reflechir a des variables type matin/soir etc...
        return agg_db




symbol = 'BNBBTC'
c = BuildDatabase(api_key, api_secret, symbol)
db = c.start_data_extract(6000)

db = c.create_modeling_database('2s')

db = pd.DataFrame(db['price'])
db.plot()
