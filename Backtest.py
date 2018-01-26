from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Backtest1:
    """
    Class for a Backtest with y and y_hat provided.
    """

    # TODO : Description of this class:
    """

    """

    def __init__(self):
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
        self.gain = 0  # gain of the strategy
        self.carry = []  # list of long position with the price of the asset or None if we don't have it

    def calculate_win(self, y, y_hat, thresh, transaction_fees=0.005, buy_fees=0.005, sell_fees=0.005):
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
        nbAchat = 0
        # buy_price and sell_price are used to calculate the performance
        # target_price and stop_price are determined so that we cut off our loss or our gain

        for i in range(len(y)):
            price = y[i]
            predicted_price = y_hat[i]
            # security : if we have a course that is 0 we have a problem
            if price == 0:
                print("We've got a problem !! price is 0 !!")
            # we have the asset
            elif long:
                self.carry.append(price)

                # we have reached our target
                if price > target_price:

                    # we keep our asset or we sell it
                    if (predicted_price - price) / price > thresh:
                        # new target_price and stop_price TODO Define a better stop loss and target price strategy
                        target_price = predicted_price - (predicted_price - price - thresh) / 2
                        stop_price = price - (predicted_price - price - thresh) / 2
                        print("On a atteint le target price au temps {} et on reste. Cours : {}".format(i, price))
                    else:
                        # we sell the asset
                        sell_price = price * (1 - sell_fees - transaction_fees)
                        index *= sell_price / buy_price
                        long = False
                        print("On a atteint le target price au temps {} et on vend à {}".format(i, price))
                        print("Performance : {0:10.2f} % ".format((sell_price / buy_price-1)*100, 2))
                        print('______________\n')

                # we have reached the stop loss price
                elif price < stop_price:
                    # we sell the asset
                    sell_price = price * (1 - sell_fees - transaction_fees)
                    index *= sell_price / buy_price
                    long = False
                    print("On a atteint le stop loss au temps {} et on vend à {}".format(i, price))
                    print("Performance : {0:10.2f} % ".format((sell_price / buy_price-1)*100, 2))
                    print('______________\n')

                    # we don't have the asset and we will buy it
            elif (predicted_price - price) / price > thresh:
                target_price = predicted_price - (predicted_price - price - thresh) / 2
                stop_price = price - (predicted_price - price - thresh) / 2
                long = True
                buy_price = price * (1 + buy_fees + transaction_fees)
                self.carry.append(price)
                print("On achete a {} au temps {}".format(price, i))
                nbAchat += 1

            else:
                self.carry.append(None)

        # in the end of the test, if we still have the asset, we sell it
        if long:
            sell_price = price * (1 - sell_fees - transaction_fees)
            index *= sell_price / buy_price
            print("On a atteint la fin des data. Cours : {}".format(y[i]))
            print("Performance : {0:10.2f} % ".format((sell_price / buy_price-1)*100, 2))
            print('______________\n')

        print("Nombre d'achat : {}".format(nbAchat))
        self.gain = index / 100 - 1
        print("Performance de la strategie : {0:10.2f} % ".format(self.gain * 100, 2))
        print('______________')
        print('______________\n')
        return self.gain

    # Tests


transaction_fees = 0.001
buy_fees = 0.005
sell_fees = 0.005
thresh = (transaction_fees + buy_fees + sell_fees) * 3

# Tests 1 check a winning strategy
y1 = np.array([1, 2, 3])
y_hat1 = np.array([3, 3, 3])
b1 = Backtest1()
Ind1 = b1.calculate_win(y=y1, y_hat=y_hat1, thresh=thresh, transaction_fees=transaction_fees, buy_fees=buy_fees,
                        sell_fees=sell_fees)


# Tests 2 check a losing strategy
y2 = np.array([3, 2, 1])
y_hat2 = np.array([5, 2, 1])
b2 = Backtest1()
Ind2 = b2.calculate_win(y=y2, y_hat=y_hat2, thresh=thresh, transaction_fees=transaction_fees, buy_fees=buy_fees,
                        sell_fees=sell_fees)


# Tests 3 check a neutral buying strategy (small loss due to transaction fees)
y3 = np.array([1, 1, 1])
y_hat3 = np.array([3, 1, 1])
b3 = Backtest1()
Ind3 = b3.calculate_win(y=y3, y_hat=y_hat3, thresh=thresh, transaction_fees=transaction_fees, buy_fees=buy_fees,
                        sell_fees=sell_fees)



class Backtest2:
    """
    Class for a Backtest with a time step and a T-period of carry
    """

    def __init__(self):
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
        self.gain = 0  # gain of the strategy
        self.carry = []  # list of long position with the price of the asset or None if we don't have it

    def calculate_win(self, h, h_win, h_loss, target_diff, loss_diff, again_max, wait_max, y, delta_y_hat, T,
                      transaction_fees=0.0005, buy_fees=0.005, sell_fees=0.005):
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

        # again/wait will be a parameter
        again = 0
        wait = 0
        nbAchat = 0

        # buy_price and sell_price are used to calculate the performance
        # target_price and stop_price are determined so that we cut off our loss or our gain

        for i in range(len(y)):
            # security : if we have a course that is 0 we have a problem
            price = y[i]
            predicted_price = delta_y_hat[i]

            if price == 0:
                print("We've got a problem !! price is 0 !!")
            # we have the asset
            elif long:
                self.carry.append(price)
                # we have reached our target price
                if price > target_price:

                    # we keep our asset or we sell it
                    if predicted_price > h_win:
                        # new target_price and stop_price TODO Define a better stop loss and target price strategy
                        target_price = price * (1. + target_diff)
                        stop_price = price * (1. - loss_diff)
                        t = T
                        again = 0
                        print("On a atteint le target price au temps {} et on reste. Cours : {}".format(i, price))
                    else:
                        # we sell the asset
                        sell_price = price * (1. - sell_fees - transaction_fees)
                        index *= sell_price / buy_price
                        long = False
                        t = 0
                        again = 0
                        print("On a atteint le target price au temps {} et on vend à {}".format(i, price))
                        print("Performance : {0:10.2f} % ".format((sell_price / buy_price-1)*100, 2))
                        print('______________\n')

                # we have reached the stop loss price
                elif price < stop_price:
                    # we have the right to stay in the deal
                    if (again < again_max) and predicted_price > h_loss:
                        again += 1
                        t = T
                        target_price = price * (1. + target_diff)
                        stop_price = price * (1. - loss_diff)
                        print("On a atteint le stop loss au temps {} et on attend pour vendre. Cours : {}".format(i,
                                                                                                                  price))
                    else:
                        # we sell the asset
                        sell_price = price * (1. - sell_fees - transaction_fees)
                        index *= sell_price / buy_price
                        long = False
                        t = 0
                        again = 0
                        print("On a atteint le stop loss au temps {} et on vend à {}".format(i, price))
                        print("Performance : {0:10.2f} % ".format((sell_price / buy_price-1)*100, 2))
                        print('______________\n')
                elif t == 0:
                    print("on a atteint la fin du periode de carry")
                    if wait < wait_max:
                        wait += 1
                        print("On reste dans le deal")
                    else:
                        wait = 0
                        # we sell the asset
                        sell_price = price * (1. - sell_fees - transaction_fees)
                        index *= sell_price / buy_price
                        long = False
                        t = 0
                        again = 0
                        print("Wait_max atteint on sort du deal au temps {} et on vend à {}".format(i, price))
                        print("Performance : {0:10.2f} % ".format((sell_price / buy_price-1)*100, 2))
                        print('______________\n')

                else:
                    # nothing happen, we decrement t
                    t -= 1

            # we don't have the asset and we will buy it
            elif predicted_price > h:
                target_price = price * (1. + target_diff)
                stop_price = price * (1. - loss_diff)
                long = True
                buy_price = price * (1. + buy_fees + transaction_fees)
                t = T
                print("On achete a {} au temps {}".format(price, i))
                nbAchat += 1
                self.carry.append(price)
            else:
                self.carry.append(None)

        # in the end of the test, if we still have the asset, we sell it
        if long:
            sell_price = price * (1 - sell_fees - transaction_fees)
            index *= sell_price / buy_price
            print("On a atteint la fin des data. Cours : {}".format(y[i]))
            print("Performance : {0:10.2f} % ".format((sell_price / buy_price-1)*100, 2))
            print('______________\n')

        print("Nombre d'achat : {}".format(nbAchat))
        self.gain = index / 100 - 1
        print("Performance de la strategie : {0:10.2f} % ".format(self.gain * 100, 2))
        print('______________')
        print('______________\n')
        return self.gain


# Tests
buy_fees = 0.0025
transaction_fees = 0.001 / 2.
sell_fees = 0.0025
T = 300  # 2s*300 = 10min
h = 2
h_win = 1
h_loss = 1
target_diff = 0.015
loss_diff = 0.012
again_max = 2
wait_max = 2

# Tests 1 check a winning strategy
# y1 = for_mouch.y.values
y1 = np.array([1, 2, 3])
# y_hat1 = for_mouch.y_thre_hat.values
y_hat1 = np.array([3, 3, 3])
b1 = Backtest2()
Ind1 = b1.calculate_win(h=h, h_win=h_win, h_loss=h_loss, target_diff=target_diff, loss_diff=loss_diff,
                        again_max=again_max, wait_max=wait_max, y=y1, delta_y_hat=y_hat1, T=T,
                        transaction_fees=transaction_fees, buy_fees=buy_fees, sell_fees=sell_fees)


# Tests 2 check a losing strategy
y2 = np.array([3, 2, 1])
y_hat2 = np.array([5, 0, 0])
b2 = Backtest2()
Ind2 = b2.calculate_win(h=h, h_win=h_win, h_loss=h_loss, target_diff=target_diff, loss_diff=loss_diff,
                        again_max=again_max, wait_max=wait_max, y=y2, delta_y_hat=y_hat2, T=T,
                        transaction_fees=transaction_fees,
                        buy_fees=buy_fees, sell_fees=sell_fees)


# Tests 3 check a neutral buying strategy (small loss due to transaction fees)
y3 = np.array([1, 1, 1])
y_hat3 = np.array([3, 0, 0])
b3 = Backtest2()
Ind3 = b3.calculate_win(h=h, h_win=h_win, h_loss=h_loss, target_diff=target_diff, loss_diff=loss_diff,
                        again_max=again_max, wait_max=wait_max, y=y3, delta_y_hat=y_hat3, T=T,
                        transaction_fees=transaction_fees,
                        buy_fees=buy_fees, sell_fees=sell_fees)

