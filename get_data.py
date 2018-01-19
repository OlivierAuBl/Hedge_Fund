from binance.client import Client
import pandas as pd
import matplotlib.pyplot as pt


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
        last_id = self.client.get_recent_trades(symbol=self.symbol, limit=1)[0]['id']
        start_id = last_id - nrows + 1
        print("Data requested to the API from Id = {} to Id = {}".format(start_id, last_id))
        if nrows % 500 == 0:
            num_ite = nrows/500
        else:
            raise AttributeError('Fuck you, you have to put a multiple of 500 for nrows, read the doc next time.')

        for i in range(start_id, last_id, 500):
            print(i)
            extract_i = self.client.get_historical_trades(symbol=self.symbol, limit=500, fromId=i)
            print(extract_i)
            self.database = pd.concat([self.database, pd.DataFrame(extract_i)])

        return self.database


    def create_modeling_database(self):
        self.database["date"] = pd.to_datetime(self.database['time'], unit='ms')
        # todo: aggregate each 10 seconds and build new features !
        return


