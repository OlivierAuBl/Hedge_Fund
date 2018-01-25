from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as pt

"""
TODO : Description of this class



"""


class Backtest1:
    """
    Class for a Backtest with y and y_hat provided.
    """

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
b1 = Backtest1(y=y1, y_hat=y_hat1, thresh=thresh, transaction_fees=transaction_fees, buy_fees=buy_fees,
               sell_fees=sell_fees)
Ind1 = b1.calculate_win
print(Ind1)

# Tests 2 check a losing strategy
y2 = np.array([3, 2, 1])
y_hat2 = np.array([5, 2, 1])
b2 = Backtest1(y=y2, y_hat=y_hat2, thresh=thresh, transaction_fees=transaction_fees, buy_fees=buy_fees,
               sell_fees=sell_fees)
Ind2 = b2.calculate_win
print(Ind2)

# Tests 3 check a neutral buying strategy (small loss due to transaction fees)
y3 = np.array([1, 1, 1])
y_hat3 = np.array([3, 1, 1])
b3 = Backtest1(y=y3, y_hat=y_hat3, thresh=thresh, transaction_fees=transaction_fees, buy_fees=buy_fees,
               sell_fees=sell_fees)
Ind3 = b3.calculate_win
print(Ind3)


class Backtest2:
    """
    Class for a Backtest with a time step and a T-period of carry
    """

    def __init__(self, y, delta_y_hat, T, transaction_fees=0.005, buy_fees=0.005, sell_fees=0.005):
        """
        Constructor
        :param y: double series
        real series
        :param delta_y_hat: double series
        predicted series variations
        :param T: int/long
        period of time before we balance our portfolio
        :param transaction_fees: double
        transaction fees as a %
        :param buy_fees: double
        bid/ask fees as a %
        :param sell_fees: double
        bid/ask fees as a %
        """
        self.y = y
        self.delta_y_hat = delta_y_hat
        self.T = T
        self.transaction_fees = transaction_fees
        self.buy_fees = buy_fees
        self.sell_fees = sell_fees

    def calculate_win(self, h, h_win, h_loss, target_diff, loss_diff, again_max):
        """
        Function used to calculate the performance of a strategy as a %
        :param h: double threshold for buying ang getting long
        :param h_win: double threshold for staying long after we hit the variation target
        :param h_loss: double threshold for staying long after we hit the stop loss floor
        :param target_diff: double difference for the target price as a %
        :param loss_diff: double difference for the stop loss price as a %
        :param again_max: int number of a maximum period of time for waiting
        :return:
            return the performance of a strategy as a %
        """
        # the index is the virtual value of our portfolio. Initially we got 100% of our money
        index = 100

        # long is true if we possess the asset
        long = False

        # t is the remaining time before we balance our portfolio
        t = 0

        # again will be a parameter
        again = 0

        # buy_price and sell_price are used to calculate the performance
        # target_price and stop_price are determined so that we cut off our loss or our gain

        for i in range(len(self.y)):
            # security : if we have a course that is 0 we have a problem
            if self.y[i] == 0:
                print("We've got a problem !! price is 0 !!")
            # we have the asset
            elif long:

                # we have reached our target price or our carry period
                if (self.y[i] > target_price) or (t == 0):

                    # we keep our asset or we sell it
                    if self.delta_y_hat[i] > h_win:
                        # new target_price and stop_price TODO Define a better stop loss and target price strategy
                        target_price = self.y[i] * (1 + target_diff)
                        stop_price = self.y[i] * (1 - loss_diff)
                        t = self.T
                        again = 0
                    else:
                        # we sell the asset
                        sell_price = self.y[i] * (1 - self.sell_fees) * (1 - self.transaction_fees)
                        index *= sell_price / buy_price
                        long = False
                        t = 0
                        again = 0

                # we have reached the stop loss price
                elif self.y[i] < stop_price:
                    # we have the right to stay in the deal
                    if (again < again_max) and self.delta_y_hat[i] > h_loss:
                        again += 1
                        t = self.T
                        target_price = self.y[i] * (1 + target_diff)
                        stop_price = self.y[i] * (1 - loss_diff)
                    else:
                        # we sell the asset
                        sell_price = self.y[i] * (1 - self.sell_fees) * (1 - self.transaction_fees)
                        index *= sell_price / buy_price
                        long = False
                        t = 0
                        again = 0
                else:
                    # nothing happen, we decrement t
                    t -= 1

            # we don't have the asset and we will buy it
            elif self.delta_y_hat[i] > h:
                target_price = self.y[i] * (1 + target_diff)
                stop_price = self.y[i] * (1 - loss_diff)
                long = True
                buy_price = self.y[i] * (1 + self.buy_fees) * (1 + self.transaction_fees)
                t = self.T

        # in the end of the test, if we still have the asset, we sell it
        if long:
            sell_price = self.y[i] * (1 - self.sell_fees) * (1 - self.transaction_fees)
            index *= sell_price / buy_price

        return index / 100 - 1


# Tests
transaction_fees = 0.005
buy_fees = 0.005
sell_fees = 0.005
T = 300  # 2s*300 = 10min
h = 1
h_win = 1
h_loss = 1.2
target_diff = 0.05
loss_diff = 0.05
again_max = 5

# Tests 1 check a winning strategy
y1 = np.array([1, 2, 3])
y_hat1 = np.array([3, 3, 3])
b1 = Backtest2(y=y1, delta_y_hat=y_hat1, T=T, transaction_fees=transaction_fees, buy_fees=buy_fees, sell_fees=sell_fees)
Ind1 = b1.calculate_win(h=h, h_win=h_win, h_loss=h_loss, target_diff=target_diff, loss_diff=loss_diff, again_max=again_max)
print(Ind1)

# Tests 2 check a losing strategy
y2 = np.array([3, 2, 1])
y_hat2 = np.array([5, 0, 0])
b2 = Backtest2(y=y2, delta_y_hat=y_hat2, T=T, transaction_fees=transaction_fees, buy_fees=buy_fees, sell_fees=sell_fees)
Ind2 = b2.calculate_win(h=h, h_win=h_win, h_loss=h_loss, target_diff=target_diff, loss_diff=loss_diff, again_max=again_max)
print(Ind2)

# Tests 3 check a neutral buying strategy (small loss due to transaction fees)
y3 = np.array([1, 1, 1])
y_hat3 = np.array([3, 0, 0])
b3 = Backtest2(y=y3, delta_y_hat=y_hat3, T=T, transaction_fees=transaction_fees, buy_fees=buy_fees, sell_fees=sell_fees)
Ind3 = b3.calculate_win(h=h, h_win=h_win, h_loss=h_loss, target_diff=target_diff, loss_diff=loss_diff, again_max=again_max)
print(Ind3)
