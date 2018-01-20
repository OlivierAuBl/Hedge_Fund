from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as pt

"""
TODO : Description of this class



"""


class Backtest:

    def __init__(self, y, y_hat, thresh, transaction_fees=0.005, buy_fees=0.005, sell_fees=0.005):
        """
        Constructor
        :param y:
        real series
        :param y_hat:
        predicted series
        :param thresh: double
        threshold as a %
        :param transaction_fees: double
        transaction fees as a %
        :param buy_fees: double
        bid/ask fees as a %
        :param sell_fees: double
        bid/ask fees as a %
        """
        self.y = y
        self.y_hat = y_hat
        self.thresh = thresh
        self.transaction_fees = transaction_fees
        self.buy_fees = buy_fees
        self.sell_fees = sell_fees

    @property
    def calculate_win(self):
        """
        Function used to calculate the performance of a strategy as a %
        :param none

        :return:
            return the performance of a strategy as a %
        """
        # the index is the virtual value of our portfolio. Initially we got 100% of our money
        index = 100

        # long is true if we possess the asset
        long = False

        # buy_price and sell_price are used to calculate the performance
        # target_price and stop_price are determined so that we cut off our loss or our gain

        for i in range(len(self.y)):
            # security : if we have a course that is 0 we have a problem
            if self.y[i] == 0:
                print("We've got a problem !! price is 0 !!")
            # we have the asset
            elif long:

                # we have reached our target
                if self.y[i] > target_price:

                    # we keep our asset or we sell it
                    if (self.y_hat[i] - self.y[i]) / self.y[i] > self.thresh:
                        # new target_price and stop_price TODO Define a better stop loss and target price strategy
                        target_price = self.y_hat[i] - (self.y_hat[i] - self.y[i] - self.thresh) / 2
                        stop_price = self.y[i] - (self.y_hat[i] - self.y[i] - self.thresh) / 2
                    else:
                        # we sell the asset
                        sell_price = self.y[i] * (1 - self.sell_fees) * (1 - self.transaction_fees)
                        index *= sell_price / buy_price
                        long = False

                # we have reached the stop loss price
                elif self.y[i] < stop_price:
                    # we sell the asset
                    sell_price = self.y[i] * (1 - self.sell_fees) * (1 - self.transaction_fees)
                    index *= sell_price / buy_price
                    long = False

            # we don't have the asset and we will buy it
            elif (self.y_hat[i] - self.y[i]) / self.y[i] > thresh:
                target_price = self.y_hat[i] - (self.y_hat[i] - self.y[i] - self.thresh) / 2
                stop_price = self.y[i] - (self.y_hat[i] - self.y[i] - self.thresh) / 2
                long = True
                buy_price = self.y[i] * (1 + self.buy_fees) * (1 + self.transaction_fees)

        # in the end of the test, if we still have the asset, we sell it
        if long:
            sell_price = self.y[i] * (1 - self.sell_fees) * (1 - self.transaction_fees)
            index *= sell_price / buy_price

        return index / 100 - 1


# Tests
transaction_fees = 0.005
buy_fees = 0.005
sell_fees = 0.005
thresh = (transaction_fees + buy_fees + sell_fees) * 3

# Tests 1 check a winning strategy
y1 = np.array([1, 2, 3])
y_hat1 = np.array([3, 3, 3])
b1 = Backtest(y=y1, y_hat=y_hat1, thresh=thresh, transaction_fees=transaction_fees, buy_fees=buy_fees,
              sell_fees=sell_fees)
Ind1 = b1.calculate_win
print(Ind1)

# Tests 2 check a losing strategy
y2 = np.array([3, 2, 1])
y_hat2 = np.array([5, 2, 1])
b2 = Backtest(y=y2, y_hat=y_hat2, thresh=thresh, transaction_fees=transaction_fees, buy_fees=buy_fees,
              sell_fees=sell_fees)
Ind2 = b2.calculate_win
print(Ind2)

# Tests 3 check a neutral buying strategy (small loss due to transaction fees)
y3 = np.array([1, 1, 1])
y_hat3 = np.array([3, 1, 1])
b3 = Backtest(y=y3, y_hat=y_hat3, thresh=thresh, transaction_fees=transaction_fees, buy_fees=buy_fees,
              sell_fees=sell_fees)
Ind3 = b3.calculate_win
print(Ind3)
