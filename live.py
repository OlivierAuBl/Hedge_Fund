from get_data import BuildDatabase
import brouillon
import pandas as pd
import time

"""
1/ get a starting dataset to have all variables
2/ each t_period seconds, create a mini db 
3/ predict from model
"""

class Live:

    def __init__(self, t_interval, model, symbol='BNBBTC'):
        """

        :param t_interval: in seconds, t choosen during/for aggregation
        :param model: needs to have method model.predict returning a np.array()
        """
        self.t_interval = t_interval
        self.model = model
        self.symbol = symbol
        symbol = 'BNBBTC'
        self.c = BuildDatabase(brouillon.api_key, brouillon.api_secret, self.symbol)
        #initialize database
        self.db = pd.DataFrame()
        self.is_data_flowing = False # Boolean detecting if we are currently scraping data through the api


        self.last_id_scrapped = 0

    def start_dataflow(self):
        # starting dataset:
        self.is_data_flowing = True
            db = self.c.start_data_extract(10*500)

        while self.is_data_flowing:
            print("Update")
            time.sleep(self.t_interval)  # Delay for t_interval seconds.

            new_data = self.c.create_modeling_database(agg_lvl='{}s'.format(int(self.t_interval)))

            print model.predict(new_data)

    def stop_dataflow(self):
        self.is_data_flowing = False #todo:  ca va pas


# Testing Class:
class ModelTest:
    def __init__(self, gbm_model, target_name, weight_name, rdm10_name):
        self.gbm_model = gbm_model
        self.target_name = target_name
        self.weight_name =weight_name
        self.rdm10_name = rdm10_name

    def predict(self, db_to_predict):#, gbm_model, db_to_predict, target_name, weight_name, rdm10_name):
        Xtest = db_to_predict.drop([target_name, weight_name, rdm10_name], inplace=False, axis=1)
        ytest = db_to_predict[target_name]
        wtest = db_to_predict[weight_name]
        xgtest = xgb.DMatrix(Xtest.values, label=ytest.values, weight=wtest.values)
        preds = self.gbm_model.predict(xgtest)
        return preds


model = ModelTest(gbm_model, target_name, weight_name, rdm10_name)
model.predict(db_training)
live = Live(5, model, symbol='BNBBTC')
live.start_dataflow()