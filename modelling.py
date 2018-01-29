import matplotlib.pyplot as plt
from get_data import BuildDatabase
import numpy as np
import pandas as pd
import xgboost as xgb
from xgb_utils import bayes_gridsearch
from xgb_utils import predict_on_holdout
import brouillon


db = pd.read_csv('data_test/raw_db_1M_260118.csv').drop('Unnamed: 0', axis=1)

build = BuildDatabase(brouillon.api_key, brouillon.api_secret, '5s')
db = build.create_modeling_database('5s', db)


########################################################################################################################
#                                      ------------------- XGB -----------------
########################################################################################################################

class ModelXGB():

    def __init__(self, db_train, db_test, t_interval, percentage_rise=2.0):
        self.db_train = db_train.copy(deep=True)
        self.db_test = db_test.copy(deep=True)

        # db_train, db_test = db_train.copy(deep=True), db_test.copy(deep=True)
        self.target_name = 'y'
        self.weight_name = 'w'
        self.rdm10_name = 'rdm10'
        self.db_train['w'], self.db_train['rdm10'] = np.ones(len(db_train)), np.ones(len(db_train))
        self.db_test['w'], self.db_test['rdm10'] = np.ones(len(db_test)), np.ones(len(db_test))
        self.db_train.drop('weekday_name', axis=1, inplace=True) #todo: better, detect object columns and impact code
        self.db_test.drop('weekday_name', axis=1, inplace=True)
        self.categorical_columns = []
        self.model = None
        print('Preparing target for Train...')
        self.db_train = self.prepare_database(self.db_train, t_interval, percentage_rise)
        print('Preparing target for test...')
        self.db_test = self.prepare_database(self.db_test, t_interval, percentage_rise)

    def prepare_database(self, database, t_interval, percentage_rise=2.0, plot=False):
        price = database['price'].values
        new_target = np.zeros(len(price))

        for i, p in enumerate(price): #todo: find a more efficient way to do that
            price_normalized = price / p
            window_price = price_normalized[i:i + t_interval]

            if max(window_price) > (1. + percentage_rise / 100.):
                new_target[i] = 1.

            for k in [2, 3, 4]:
                if max(window_price) > (1. + k * percentage_rise / 100.):
                    new_target[i] = k

        print('{}% of the database is not null'.format(100.*(new_target > 0).sum() / float(len(new_target))))

        if plot:
            import matplotlib.pyplot as plt
            plt.hist(new_target)

        return database.assign(y=new_target)

    def plot_distrib_train(self):
        import matplotlib.pyplot as plt
        try:
            plt.hist(self.db_train['y'])
        except:
            print('You need to prepare database first. see ModelXGB.prepare_database()')

    def find_parameters(self, n_Xval=5, num_rounds=500, n_points=10, kappa=5, dict_bounds=None):
        """
        Perform a Bayesian Gridsearch to search for the best meta-parameters
        :param n_Xval:
        :param num_rounds:
        :param n_points:
        :param kappa:
        :param dict_bounds:
        :return:
        """
        #n_Xval = 4 # num of folds for X-validation
        #num_rounds = 400  # num ite XGBoost
        #n_points = 6 # num points Bayesian gridsearch
        if dict_bounds is None:
            dict_bounds = {'eta': (0.01, 0.05), 'max_depth': (3, 5), 'min_child_weight': (1, 3), 'gamma': (0., 0.2),
                           'lambda_': (0.1, 1.5), 'colsample_bytree': (0.6, 0.9), 'subsample': (0.6, 0.9)}

        bg, bo = bayes_gridsearch(self.db_train,
                                  self.target_name,
                                  self.categorical_columns,
                                  self.weight_name,
                                  dict_bounds,
                                  num_rounds,
                                  kappa=kappa,
                                  n_Xval=n_Xval,
                                  n_points=n_points,
                                  nthread=8,
                                  eval_metric='logloss',
                                  objective='count:poisson',  # "reg:gamma"
                                  outfile=None)


        bg.sort_values('gini_test', ascending=False, inplace=True)
        print(bg)
        return bg

    def train(self, bg):

        params_line = bg
        # Build GBM on train, apply to the whole database
        path_plot_save = '/Users/axa/Documents/Git/Hedge_Fund/plots/'
        db_tot, gbm_model = predict_on_holdout(self.db_train, self.db_test, self.target_name, self.weight_name, self.rdm10_name,
                                               self.categorical_columns, params_line, objective='count:poisson', nthread=8,
                                               path_plot_save=path_plot_save, add_log=False)

        self.model = gbm_model
        return gbm_model

    def predict(self, db_to_predict):
        Xtest = db_to_predict.drop([self.target_name, self.weight_name, self.rdm10_name], inplace=False, axis=1, errors='ignore')
        #ytest = db_to_predict[self.target_name]
        #wtest = db_to_predict[self.weight_name]
        xgtest = xgb.DMatrix(Xtest.values)#, label=ytest.values, weight=wtest.values)
        if self.model is not None:
            preds = self.model.predict(xgtest)
            return preds
        else:
            print('You need to train the model before predicting. (see ModelXGB.train())')

    def plot(self, treshold_buy):

        future = self.predict(self.db_train)
        price = self.db_train['price'].copy(deep=True)
        for i in range(len(future)):
            if future[i] < treshold_buy:
                future[i] = None
                price[i] = None
        to_plot = self.db_train.assign(y=self.db_train.price.values, y_thre_hat=future)[['y', 'y_thre_hat']]

        plt.plot(self.db_train['price'].values)
        plt.plot(to_plot.index, (to_plot.y_thre_hat.values > treshold_buy) * price, 'ro', markersize=1)
        plt.show()



#Create train / test
df = db.copy(deep=True)
part_db = 0.2
l = len(df)
int_stop = int(l*(1 - 0.2))
db_train = df.iloc[0:int_stop, :]
db_test = df.iloc[int_stop:, :]

# test:
# Prepare dataset by creating the good target (here, will there be a 1.5% rise in the next 120*5s = 10 minutes ?)
m = ModelXGB(db_train, db_test, t_interval=120, percentage_rise=1.5)
m.plot_distrib_train()

# Compute Bayesian gridsearch to find parameters (Directly load the BayGrid to avoid rerunning the model):
# bg = m.find_parameters(n_Xval=5, num_rounds=500, n_points=10, kappa=5, dict_bounds=None)
bg = pd.read_csv('data_test/params_bg_5s_10min.csv').drop('Unnamed: 0', axis=1)
# Train the model with the parameters:
gbm_model = m.train(bg)

# We can predict on test:
m.predict(db_test.drop('weekday_name', axis=1))

#plot:
m.plot(treshold_buy=1.5)


