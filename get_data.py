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
            print(i)
            extract_i = self.client.get_historical_trades(symbol=self.symbol, limit=500, fromId=i)
            print(extract_i)
            self.database = pd.concat([self.database, pd.DataFrame(extract_i)])

        return self.database


    def create_modeling_database(self):

        # Changing the awful time display into a date
        self.database["date"] = pd.to_datetime(self.database['time'], unit='ms')
        # todo: aggregate each 10 seconds and build new features !
        return


