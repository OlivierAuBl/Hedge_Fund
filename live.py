from binance.client import Client
from get_data import BuildDatabase
import brouillon
import pandas as pd
from time import sleep


class Live:

    def __init__(self, t_interval, model, symbol='BNBBTC'):
        """

        :param t_interval: in seconds, t choosen during/for aggregation
        :param model: needs to have method model.predict returning a np.array()
        """
        self.t_interval = t_interval
        self.model = model
        self.symbol = symbol
        self.client = Client(brouillon.api_key, brouillon.api_secret)
        #initialize database
        self.db = pd.DataFrame()
        self.last_id_scrapped = 0

        self.carry = []  # list of long position with the price of the asset or None if we don't have it


    def start_dataflow(self):
        # starting dataset:

        aggregator = BuildDatabase(brouillon.api_key, brouillon.api_secret, symbol)
        data_init = aggregator.start_data_extract(20000)
        last_id = data_init['id'].max()
        # data_init_agg = aggregator.create_modeling_database(agg_lvl='{}s'.format(self.t_interval), database=data_init)

        threshold = 1

        while True:
            sleep(5)
            # Update one step:
            len_max_dataset = 20000
            data_live = pd.DataFrame(self.client.get_historical_trades(symbol=symbol, fromId=last_id + 1))
            updated_db = pd.concat([data_init, data_live])
            last_id = updated_db['id'].max()
            # delete points too old not to increase memory storage.
            if len(updated_db) > len_max_dataset:
                updated_db = updated_db.iloc[-len_max_dataset:]
            # Aggregate database
            updated_agg_db = aggregator.create_modeling_database(agg_lvl='{}s'.format(self.t_interval), database=updated_db)

            line_to_predict = updated_agg_db.iloc[-1:]

            # Use the model to get the prediction for the next minutes
            preds = self.model.predict(line_to_predict.drop('weekday_name', axis=1))

            pred = preds[0]  # maybe put a smooth function here. (take the mean of the last points ?
            print('Prediction at time {}: {}'.format(line_to_predict.index[0], pred))

            # Buy / Sell:
            self.bot_action_scenario(pred, threshold_buy=1)

    #def function bot() that take the decision to buy or sell: #todo: @Mouch
    def bot_action_scenario(self, predicted_price, threshold_buy):
        if predicted_price > threshold_buy:
            print("j'acheeeeeeete ! ")



l = Live(5, model=m, symbol='BNBBTC')

l.start_dataflow()


