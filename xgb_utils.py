import pandas as pd
from sklearn.model_selection import KFold
from operator import itemgetter
from collections import Counter
import numpy as np
import xgboost as xgb
import itertools
from bayes_opt import BayesianOptimization
import os
import matplotlib
import xgbfir
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#push test


def weighted_avg_and_var(values, weights, unbiased=True):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    if unbiased:
        variance = np.sum(weights*(values-average)**2)/(np.sum(weights)-np.sum(weights**2)/np.sum(weights))
    else:
        variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise

    return average, variance


def check_deviance(y, y_pred, weight=None):
    """
    Robust checks to run at beginning of deviance
    """
    if isinstance(y_pred, (np.floating, float)):
        y_pred = np.repeat(y_pred, y.shape[0])
    assert y.shape[0] == y_pred.shape[0], "y and y_pred must have the same size"
    if weight is not None:
        assert weight.shape[0] == y.shape[0], "weight and y do not have same shape"
    return y_pred


def binomial_deviance(y, y_pred, weight=None):
    """
    Variance for the binomial model

    Parameters
    ----------

    y : ndarray
        array containing the TRUE response (either 0 or 1)
    y_pred : ndarray
        array containing the predicted probabilities by the model
    weight : ndarray, optional
        array containing the weight (default 1)

    Returns
    -------
    ndarray
        computed deviance

    """
    y_pred = check_deviance(y, y_pred, weight=weight)
    deviance_vector = - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    if weight is not None:
        deviance_vector = np.dot(weight, deviance_vector)
    return 2 * np.sum(deviance_vector)


def gamma_deviance(y, y_pred, weight=None):
    """
    Deviance function for gamma model.
    Exactly the same as the one implement in Emblem.

    Parameters
    ----------

    y : ndarray
        array containing the TRUE response (either 0 or 1)
    y_pred : ndarray
        array containing the predicted probabilities by the model
    weight : ndarray, optional
        array containing the weight (default 1)

    Returns
    -------
    ndarray
        computed deviance


    """
    y_pred = check_deviance(y, y_pred, weight=weight)

    deviance_vector = -np.log(y / y_pred) + (y - y_pred) / y_pred
    if weight is not None:
        deviance_vector = np.dot(weight, deviance_vector)

    return 2 * np.sum(deviance_vector)


def poisson_deviance(y, y_pred, weight=None):
    """
    Deviance function for the poisson model.

    Parameters
    ----------

    y : ndarray
        array containing the TRUE response (either 0 or 1)
    y_pred : ndarray
        array containing the predicted probabilities by the model
    weight : ndarray, optional
        array containing the weight (default 1)

    Returns
    -------
    ndarray
        computed deviance

    """
    y_pred = check_deviance(y, y_pred, weight=weight)

    bool_zeros = y != 0
    deviance_vector = np.zeros(y.shape[0])
    deviance_vector[bool_zeros] = (
            y[bool_zeros] * np.log(y[bool_zeros] / y_pred[bool_zeros]) - y[bool_zeros] + y_pred[bool_zeros])
    deviance_vector[~bool_zeros] = - y[~bool_zeros] + y_pred[~bool_zeros]
    if weight is not None:
        deviance_vector = np.dot(weight, deviance_vector)

    return 2 * np.sum(deviance_vector)


def gaussian_deviance(y, y_pred, weight=None):
    """
    Deviance function for the gaussian/least squares model.

    Parameters
    ----------

    y : ndarray
        array containing the TRUE response (either 0 or 1)
    y_pred : ndarray
        array containing the predicted probabilities by the model
    weight : ndarray, optional
        array containing the weight (default 1)

    Returns
    -------
    ndarray
        computed deviance

    """
    y_pred = check_deviance(y, y_pred, weight=weight)
    deviance_vector = np.square(y - y_pred)
    if weight is not None:
        deviance_vector = np.dot(weight, deviance_vector)
    return 0.5 * np.sum(deviance_vector)


def gaussian_pseudo_r2(y, y_pred, weight=None):
    """
    PseudoR2 for a Normal model.

    Parameters
    ----------

    y : ndarray
        array containing the TRUE response (either 0 or 1)
    y_pred : ndarray
        array containing the predicted probabilities by the model
    weight : ndarray, optional
        array containing the weight (default 1)

    Returns
    -------
    ndarray
        computed pseudo_R2

    Notes
    -----
    Pseudo R2 is defined as 1 - (deviance(y,y_pred,weight)}deviance(y,mu,weight)
    where mu is the weighted mean of y

    """
    return 1 - (gaussian_deviance(y, y_pred, weight)
                / gaussian_deviance(y, np.ones(len(y)) * np.average(y, weights=weight), weight))


def poisson_pseudo_r2(y, y_pred, weight=None):
    """
    PseudoR2 for the Poisson model.

    Parameters
    ----------

    y : ndarray
        array containing the TRUE response (either 0 or 1)
    y_pred : ndarray
        array containing the predicted probabilities by the model
    weight : ndarray, optional
        array containing the weight (default 1)

    Returns
    -------
    ndarray
        computed pseudo_R2

    Notes
    -----
    Pseudo R2 is defined as 1 - (deviance(y,y_pred,weight)}deviance(y,mu,weight)
    where mu is the weighted mean of y

    """
    return 1 - (poisson_deviance(y, y_pred, weight)
                / poisson_deviance(y, np.ones(len(y)) * np.average(y, weights=weight), weight))


def gamma_pseudo_r2(y, y_pred, weight=None):
    """
    Pseudo R2 for gamma model

    Parameters
    ----------

    y : ndarray
        array containing the TRUE response (either 0 or 1)
    y_pred : ndarray
        array containing the predicted probabilities by the model
    weight : ndarray, optional
        array containing the weight (default 1)

    Returns
    -------
    ndarray
        computed pseudo_R2

    Notes
    -----
    Pseudo R2 is defined as 1 - (deviance(y,y_pred,weight)}deviance(y,mu,weight)
    where mu is the weighted mean of y

    """
    return 1 - (gamma_deviance(y, y_pred, weight)
                / gamma_deviance(y, np.ones(len(y)) * np.average(y, weights=weight), weight))


def binomial_pseudo_r2(y, y_pred, weight=None):
    """
    PseudoR2 for the binomial model


    Parameters
    ----------

    y : ndarray
        array containing the TRUE response (either 0 or 1)
    y_pred : ndarray
        array containing the predicted probabilities by the model
    weight : ndarray, optional
        array containing the weight (default 1)

    Returns
    -------
    ndarray
        computed pseudo_R2

    Notes
    -----
    Pseudo R2 is defined as 1 - (deviance(y,y_pred,weight)}deviance(y,mu,weight)
    where mu is the weighted mean of y

    """
    return 1 - (binomial_deviance(y, y_pred, weight)
                / binomial_deviance(y, np.ones(len(y)) * np.average(y, weights=weight), weight))


def area_lorentz_fast(y, y_pred, weight=None, resolution=5000, interpolation="constant", plot=False):
    '''
    Reproduces the weighted gini of emblem

    Parameters
    ----------

    y : ndarray
        array containing the TRUE response (either 0 or 1)
    y_pred : ndarray
        array containing the predicted probabilities by the model
    weight : ndarray, optional
        array containing the weight (default 1)
    resolution : int, optional
        the number of points in the plot of the lorenz curve (default is 5000)
    interpolation: {'linear', 'constant'}, optional
        type of interpolation for the lorenz curve (default "constant")
    plot: bool, optional
        compute or not interpolation of lorenz curve(default False)

    '''
    # Comments
    # --------
    #
    # constant piecewise interpolation is useful when the number of observed ones is
    # little (to underline the breakpoints),
    # linear has nicer smoothing properties

    if interpolation not in ["linear", "constant"]:
        raise NotImplementedError("interpolation available only for linear and constant")
    if y.shape[0] != y_pred.shape[0]:
        raise ValueError("y and y_pred must have the same length")

    n_samples = y.shape[0]

    if weight is None:
        weight = np.repeat([1. / n_samples], n_samples)

    # Id of each column
    obs_col, pred_col, w_col, rank_col = (0, 1, 2, 3)

    # Order data following prediction
    ordered_data = np.column_stack((y, y_pred, weight, np.zeros(y.shape[0])))

    pred_order = np.argsort(y_pred)[::-1]
    ordered_data = ordered_data[pred_order, :]

    # Compute the rank
    ordered_data[:, rank_col] = np.cumsum(ordered_data[:, w_col]) - 1. / 2 * ordered_data[:, w_col]

    total_weight = np.sum(ordered_data[:, w_col])

    obs_sum = np.dot(ordered_data[:, w_col], ordered_data[:, obs_col])

    intermediate = ordered_data[:, 0] * ordered_data[:, 2] * ordered_data[:, 3]
    rank_obs_sum = intermediate.sum()

    # Compute the Gini coefficient
    gini = 1 - (2 / (total_weight * obs_sum)) * rank_obs_sum
    # until here, as in the old code
    if plot:
        # Determine the points to plot
        x_list = np.cumsum(ordered_data[:, w_col]) / total_weight
        y_list = np.cumsum(ordered_data[:, w_col] * ordered_data[:, obs_col]) / obs_sum
        x_interpolate = np.linspace(0, 1, num=resolution)

        #    this is for linear interpolation
        if interpolation == "linear":
            y_interpolate = np.interp(x_interpolate, x_list, y_list)
        elif interpolation == "constant":
            # this is for piecewise interpolation (better when 1s and 0s are little)
            f = interpolate.interp1d(x_list, y_list, kind='zero', bounds_error=False)
            y_interpolate = f(x_interpolate)

            # manually make 0 and 1s outside the range (bug in interp1d)
            y_interpolate[x_interpolate <= np.min(x_list)] = 0
            y_interpolate[x_interpolate >= np.max(x_list)] = 1

        # TODO: PLOT !

        return gini, y_interpolate
    else:
        return gini


def gini_emblem_fast(y, y_pred, weights=None, normalize_gini=False, verbose=False):
    # We compute Gini coefficient for the model col_score
    gini_model = area_lorentz_fast(y, y_pred, weights)
    if verbose:
        print "Gini coefficient for prediction", " without normalization:", gini_model
    # Emblem by default returns the non-normalized version of the Gini
    if normalize_gini:
        # We compute the gini coefficient for the "perfect model":
        gini_perfect_model = area_lorentz_fast(y, y, weights)
        if verbose:
            print "Gini coefficient of 'perfect' model:", gini_perfect_model

        # We normalize the Gini coefficient:
        gini = gini_model / gini_perfect_model
        if verbose:
            print "The Gini coefficient for prediction", " after normalization:", gini
        return gini

    else:
        # We don't normalize the Gini coefficient:
        return gini_model


def plot_lift_curve(y, y_pred, weight=None, n_band=10, title=None, path_plot_save='Results\\'):
    """

    Parameters
    ----------

    y : ndarray
        array containing the TRUE response (either 0 or 1)
    y_pred : ndarray
        array containing the predicted probabilities by the model
    weight : ndarray, optional
        array containing the weight (default 1)
    n_band : int, optional
        number of bands (default 10)
    title: str
        title of the plot, if None, plot is not saved.
    path_plot_save: str
        file path where the plot will be saved.

    Returns
    -------

    """
    if weight is None:
        weight = np.ones(y.shape[0])

    d = {'pred': list(y_pred), 'obs': list(y), 'weights': list(weight)}
    d = pd.DataFrame(d)
    d = d.dropna(subset=['obs', 'pred'])
    d = d.sort_values('pred', ascending=True)
    l = len(y_pred)
    d.index = list(range(0, l))
    exp_cum = [0]
    for k in range(0, l):
        exp_cum.append(exp_cum[-1] + d.ix[k, 'weights'])
    s = exp_cum[-1]
    j = s // n_band
    m_pred, m_obs, m_weight = [], [], []
    k, k2 = 0, 0

    for i in range(0, n_band):
        k = k2
        for p in range(k, l):
            if exp_cum[p] < ((i + 1) * j):
                k2 += 1
        temp = d.ix[range(k, k2),]
        m_pred.append(sum(temp['pred'] * temp['weights']) / sum(temp['weights']))
        m_obs.append(sum(temp['obs'] * temp['weights']) / sum(temp['weights']))
        m_weight.append(temp['weights'].sum())

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    ax1.set_xlabel('Band')
    ax1.set_ylabel('Y values')
    ax2.set_ylabel('Weight')
    ax2.set_ylim([0, max(m_weight) * 3])
    ax1.plot(range(0, n_band), m_pred, linestyle='--', marker='o', color='b')
    ax1.plot(range(0, n_band), m_obs, linestyle='--', marker='o', color='r')
    # the histogram of the weigths
    ax2.bar(range(0, n_band), m_weight, color='yellow', alpha=0.2)
    ax1.legend(labels=['Predicted', 'Observed'], loc=2)
    if title is not None:
        fig.savefig(path_plot_save + title + '.png', bbox_inches='tight')


def create_sampled_db(db, size, target_name, weight_name=None, verbose=True, frequent_target=0):
    """
    Good models can be obtained by training on smaller modified database, if the target represents
    un-balanced events (number of events with many 0, or a classification problem with a unballanced classes).
    The idea is to delete lines  bringing less information (where the target is null).
    The exposure then have to be modified in order to keep the initial frequency.

    Parameters
    ----------
    db: pandas.DataFrame
        The database to be reduced
    size: float
        Number of lines of the reduced database if >1,
        percentage of lines to keep if <1
    target_name: str
        name of the target variable column in the DataFrame
    weight_name: str
        Name of the weight column in the DataFrame
    verbose: bool, optional
        print information if Verbose = True (default = True)
    frequent_target: float, option
        the most frequent value of the target (default = 0

    Returns
    =======
    pandas.DataFrame
        The reduced database with 'size' lines and the same frequency for the target
    """
    if weight_name is None:
        weight_name, delete_weight = 'w', True
        db = db.assign(w=np.ones(len(db)))
    else:
        delete_weight = False

    positive_db, null_db = db.loc[(db[target_name] != frequent_target)].copy(), \
                           db.loc[(db[target_name] == frequent_target)].copy()
    positive_exposure, null_exposure = positive_db[weight_name].sum(), null_db[weight_name].sum()
    total_exposure = positive_exposure + null_exposure

    if size < 1:
        size = int(round(size * len(db)))

    initial_proportion = positive_exposure / total_exposure

    if verbose:
        print 'Initial database:\n - %s lines \n - %s years of exposure' % (len(db), positive_exposure)
        print 'Initial proportion (exposure) of positives: %s %%' % (100 * initial_proportion)

    num_of_lines_to_keep = size - len(positive_db)
    if num_of_lines_to_keep < 0:
        raise Exception('You are asking for a too small database, there won\'t be any null lines left')

    # keeping only the good of lines
    null_db_reduced = null_db.sample(num_of_lines_to_keep)
    change_of_proportion = null_exposure / null_db_reduced[weight_name].sum()
    #  Updating weights according this change of proportion
    null_db_reduced[weight_name] *= change_of_proportion
    # Appending the null and positive database
    db = null_db_reduced.append(positive_db)

    if verbose:
        print '\nNew database:\n - %s lines\n - %s years of exposure (artificially increased)' % (size, db[weight_name].sum())
        print 'This database is built to have the same proportion (exposure) of positives'

    if delete_weight:
        db.drop(weight_name, inplace=True, axis=1)
    return db.sort_index()


def split_train_holdout(database, rdm10_name, train=None, holdout=None, drop_rdm=False):
    """
    Split according to VAL_random10 and delete it in train and test

    Parameters
    ----------
    rdm10_name: str
        name of the random variable
    database: pandas.DataFrame
        database to split into train/test
    train: int list
        index you want to keep for training
    holdout: int list
        index to keep for validation set
    drop_rdm: bool
        If True, drops the column rdm10_name

    Returns
    -------
    :return: pandas.DataFrame, pandas.DataFrame
        Train en test database without the variable VAL_random10


    """

    if (holdout is None or train is None) and (holdout is not None or train is not None):
        raise ValueError('You\'ve defined train without holdout. Please define both or none.')
    if train is None:
        train, holdout = [1, 2, 3, 4, 5, 6, 7, 8], [9, 10]

    train_db = database[database[rdm10_name].isin(train)]
    test_db = database[database[rdm10_name].isin(holdout)]
    if drop_rdm:
        train_db = train_db.drop([rdm10_name], axis=1)
        test_db = test_db.drop([rdm10_name], axis=1)
    return train_db, test_db


def create_impact_coding(table_in_train, table_in_test, cat_cols, target, weight=None,
                         regroup_rare_modalities_threshold=0.1, remove_original_cols=False, verbose=1):
    # TODO: Make it work for classif as well (target variable as categorical variable).
    # TODO: Add a smoothing function instead pof regroup rare modalities: https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    """
    Create an impact coding of a database.
    BE CAREFUL: The target variable has to be a number (int or float)


    Parameters
    ----------

    table_in_train : DataFrame
        The train database ; it will be used to compute some values & be encoded

    table_in_test : DataFrame
        The test database ; it will use values from the train database, and will
        be encoded. Set it to None if you just want to transform a train.

    cat_cols : list of Strings
        The columns to encode

    weight : String
        The name of the weight column

    target : String
        The name of the target column

    regroup_rare_modalities_threshold : numeric
        The number of exposure necessary to create a dummy variables. If lower
        than 1, a % of the train database ; if higher, an absolute number.
        Default = 10%

    remove_original_cols : Boolean
        Should the original columns be removed after process ?

    verbose : integer
        Should it print comment (1 = few, 2 = a lot) or not (0)

    Returns
    -------

    table_in_train : DataFrame
        The encoded train database ; if inplace == False

    table_in_test : DataFrame
        The encoded train database ; if inplace == False

    impact_cols : List
        The list of the columns created

    """



    if table_in_test is None:  # in case we just want to work on a train DB, we create an empty test
        table_in_test = table_in_train[:0].copy(deep=True)
    assert set(cat_cols) < set(table_in_train.columns) and \
           set(cat_cols) < set(table_in_test.columns), \
        "!! columns of train and test should be the same !!"

    table_in_train = table_in_train.copy(deep=True)
    table_in_test = table_in_test.copy(deep=True)


    # If no weights, add a columns of ones.
    delete_weight = False
    if weight is None:
        weight, delete_weight = 'w', True
        table_in_train = table_in_train.assign(w=np.ones(len(table_in_train)))

    weight_sum = float(table_in_train[weight].sum(axis=0))

    # regroup_rare_modalities_threshold can either be a number of observation or a % of the rows....
    if regroup_rare_modalities_threshold < 1:
        regroup_rare_modalities_threshold *= weight_sum

    impact_cols = []  # This vector will contain the name of the columns created

    for colI in cat_cols:
        if verbose >= 1:
            print "impact coding column " + str(colI)
        # First, fill NA with 'OTHER'
        table_in_train[colI] = table_in_train[colI].fillna('OTHER', inplace=False)
        # We will add two columns: one with the average of the target, one with the absolute number of occurrences.
        average_target_i_name, count_col_i_name = colI + "__avg__" + target, colI + "__count"

        # Build a table with the variables of interest alone.
        sub_table_colI = table_in_train[[colI, weight, target]]
        sub_table_colI[count_col_i_name] = 1

        # Build the aggregated table:
        average_table = pd.pivot_table(table_in_train.assign(Y_target=lambda t: t[target] * t[weight],
                                                             Y_count=lambda t: t[weight]),
                                                             index=[colI],
                                                             values=['Y_count','Y_target'],
                                                             aggfunc={'Y_count': np.sum,'Y_target': np.sum})

        average_table = average_table.assign(Y_target=lambda t: t["Y_target"] / t['Y_count'])

        average_table.rename(columns={'Y_target': average_target_i_name, 'Y_count': count_col_i_name}, inplace=True)

        # For rare modality, we want to compute an "OTHER" average target value, and use it for them...
        rare_modality_table = average_table.loc[average_table[count_col_i_name] <= regroup_rare_modalities_threshold]
        if not rare_modality_table.empty:
            #  sum of all the values under the threshold:
            others_count, others_target = rare_modality_table.sum()[[count_col_i_name, average_target_i_name]]

            # If we have enough rare modalities, we will use their average target value.
            # If not, we take the whole average on the DB (not really sure this is the best way to do)
            # Maybe merge with the modality closest to the average ???? or ask ... ?
            if others_count > regroup_rare_modalities_threshold:
                avg_target_others = rare_modality_table[count_col_i_name].dot(rare_modality_table[average_target_i_name])/ \
                                    rare_modality_table[count_col_i_name].sum()
            else:
                avg_target_others = average_table[average_target_i_name].dot(average_table[count_col_i_name])/\
                                    average_table[count_col_i_name].sum()
                if verbose == 2:
                    print('Be careful, on variable: "' + colI + '", grouping of rare modalities do not reach threshold\n')
                    print('Maybe you could join these modalities manually during pre-processing with a common one.'
                          '\nSmall exposure modalities:\n' + str(rare_modality_table))
                    print('\nBig exposure modalities:\n' + str(average_table.loc[average_table[count_col_i_name] >
                                                                                regroup_rare_modalities_threshold]))

            # replace rare modalities:
            average_table.loc[average_table[count_col_i_name] <= regroup_rare_modalities_threshold, average_target_i_name]\
                = avg_target_others

        if verbose == 2:
            print average_table

        # We can merge the average target values back in the train and test DBs

        table_in_train = pd.merge(table_in_train, average_table, how="left", left_on=colI, right_index=True)
        table_in_test = pd.merge(table_in_test, average_table, how="left", left_on=colI, right_index=True)

        # If some modalities are in the test DBs but not in the aggregated DB,they will appear as NA...
        # This appends in the rares modalities that are only in the test DB
        if not rare_modality_table.empty:
            table_in_test[average_target_i_name].fillna(avg_target_others, inplace=True)
            # For count, just append the count of small modalities.
            table_in_test[count_col_i_name].fillna(others_count, inplace=True)

        # We add the new column to the list
        impact_cols = impact_cols + [average_target_i_name]
        # Finally, we remove the original column from our database
        if remove_original_cols:
            table_in_train.drop(colI, inplace=True, axis=1)
            table_in_test.drop(colI, inplace=True, axis=1)

    if delete_weight:
        table_in_train.drop(weight, inplace=True, axis=1)

    return table_in_train, table_in_test, impact_cols


def create_dummies(table_in_train, table_in_test, cat_cols, weight=None, min_nb_exp=0.1, inplace=False,
                   remove_original_cols=False, verbose=1):
    """
    Create dummy encoding (1-hot encoding) of a database.

    Parameters
    ----------
    table_in_train : DataFrame
        The train database ; it will be used to compute some values & be encoded

    table_in_test : DataFrame
        The test database ; it will use values from the train database, and will
        be encoded.

    weight : String
        The name of the weight column

    cat_cols : list of Strings
        The columns to encode

    min_nb_exp : numeric
        The number of exposure necessary to create a dummy variables. If lower
        than 1, a % of the train database ; if higher, an absolute number.
        Default = 0.1

    inplace : Boolean
        Should the process be done in-place ? if not, create a copy of the
        databases.
        Default = False

    remove_original_cols : Boolean
        Should the original columns be removed after process ?

    verbose : integer
        Should it print comment (1 = few, 2 = a lot) or not (0)

    Returns
    -------

    table_in_train : DataFrame
        The encoded train database ; if inplace == False

    table_in_test : DataFrame
        The encoded train database ; if inplace == False

    dummy_cols : List
        The list of the columns created

    """

    def series_to_bool(series_in, value_tested):
        array_out = np.zeros(len(series_in))
        array_out[(series_in == value_tested).values] = 1
        return array_out

    if not inplace:
        table_in_train = table_in_train.copy(deep=True)
        table_in_test = table_in_test.copy(deep=True)

    if table_in_test is None:  # in case we just want to work on a train DB, we create an empty test
        table_in_test = table_in_train[:0].copy(deep=True)
    assert set(cat_cols) < set(table_in_train.columns) and set(cat_cols) < set(table_in_test.columns), \
        "!! columns of train and test should be the same !!"
    # If no weights, create a column of ones.
    delete_weight = False
    if weight is None:
        weight, delete_weight = 'w', True
        table_in_train = table_in_train.assign(w=np.ones(len(table_in_train)))

    weight_sum = float(table_in_train[weight].sum(axis=0))

    # min_nb_exp can either be a number of exposure or a % of the exposure....
    if min_nb_exp < 1:
        min_nb_exp = min_nb_exp * weight_sum

    dummy_cols = []   # This vector will contain the name of the columns created

    for colI in cat_cols:
        table_in_train[colI].fillna('OTHER', inplace=False)

        list_mod = np.unique(table_in_train[colI].unique())
        if verbose == 1: print "dummies " + colI + ' ' + str(len(list_mod))

        # The Other column will contain all the "OTHER" from the original DB +
        # the rare modalities
        others_col_train = np.zeros(len(table_in_train))
        others_col_test = np.zeros(len(table_in_test))

        for modI in list_mod:
            dummies_col_i_train = series_to_bool(table_in_train[colI], modI)
            dummies_col_i_test = series_to_bool(table_in_test[colI], modI)

            if verbose == 2: print "         " + str(colI) + ' ' + str(modI) + "  :  ", str(sum(dummies_col_i_train))
            if table_in_train[weight].dot(dummies_col_i_train) > min_nb_exp and colI != "OTHER":

                # If there are enough occurrences in the train set, we create a
                # new dummy column for the current modality
                name_dummy_col_i = unicode(colI) + u'X' + unicode(modI)
                table_in_train[name_dummy_col_i] = dummies_col_i_train
                table_in_test[name_dummy_col_i] = dummies_col_i_test
                dummy_cols = dummy_cols + [name_dummy_col_i] # We add the name to the list
            else:
                # If the modality is not kept, we add it to the OTHER column
                others_col_train = others_col_train + dummies_col_i_train
                others_col_test = others_col_test + dummies_col_i_test

        # if necessary, we add a column with all the OTHERs
        if sum(others_col_train) + sum(others_col_test) > 0:
            name_dummy_others_i = unicode(colI) + u'X' + "OTHER"
            table_in_train[name_dummy_others_i] = others_col_train
            table_in_test[name_dummy_others_i] = others_col_test
            dummy_cols = dummy_cols + [name_dummy_others_i]

        # Finally, we remove the original column from our database
        if remove_original_cols:
            table_in_train.drop(colI, inplace=True, axis=1)
            table_in_test.drop(colI, inplace=True, axis=1)

    if delete_weight:
        table_in_train.drop(weight, inplace=True, axis=1)

    if not inplace:
        return table_in_train, table_in_test, dummy_cols
    else:
        return dummy_cols


def get_quantiles(vect_in, expo_in=None, nb_quantiles=100):
    """
    Rounds a vector of numerics around its quantiles.
    This pre-processing may be useful when the number of modalities of a -numeric- variable should not be too high (in
    particular in the auto-GLM algorithm, but also on a naive regression trees implementation or the bump-hunting algo).

    Parameters
    ----------
    vect_in: array_like
        Array to be rounded.
    expo_in: array_like, optional
        Exposure used to measure the quantiles.
        If None, all the elements of the array have the same weight. ( default = True)
    nb_quantiles: integer, optional
        Exposure used to measure the quantiles ( default = 100)

    Returns
    -------
    array_like
        An array containing rounded values of the vect_in array.
    """

    if expo_in is None:
        expo_in = np.ones(len(vect_in))

    quantiles_df = pd.DataFrame({"val": vect_in, "expo": expo_in})
    quantiles_df.sort_values("val", inplace=True)

    sum_expo = quantiles_df["expo"].sum()
    quantiles_df["quantile"] = np.floor(quantiles_df["expo"].cumsum() * 0.99999999 * nb_quantiles / sum_expo)

    quantiles_df["q_val"] = quantiles_df.groupby("quantile")["val"].transform(min)
    quantiles_df.sort_index(inplace=True)

    return quantiles_df["q_val"].values


def understand_smoothing_params():
    """
    Plot the functions you will be using in create_smooth_impact_coding()
    :return:
    """
    import matplotlib.pyplot as plt
    import numpy as np
    x=np.arange(0,100,0.1)
    p = lambda x, m: x / (m + x)
    plt.figure(0)
    for i in np.arange(0.25,6,0.25):
        if i < 1.6 or i%1==0:
            plt.plot(x,p(x,i), label="m = {} %".format(i))
    plt.legend()
    plt.xlabel('Percentage of total weight for modality')
    plt.ylabel('Weight given to the modality (vs prior)')
    plt.title('Simple parameter version')
    plt.grid(True)
    plt.show()

    plt.figure(1)
    p = lambda x, k, f: 1 / (1 + np.exp(-(x - k) / f))
    for i in np.arange(1, 8, 1):
            plt.plot(x, p(x, 30, i), label="f = {} %".format(i))
    plt.legend()
    plt.xlabel('Percentage of total weight for modality')
    plt.ylabel('Weight given to the modality (vs prior)')
    plt.title('Double parameter version, with k=30')
    plt.grid(True)
    plt.show()


def create_smooth_impact_coding(table_in_train, table_in_test, cat_cols, target, weight=None, smoothing_1=None,
                                smoothing_2=None, remove_original_cols=False, add_count_col=False, verbose=1):

    """
    Create a "smooth impact coding" of the categorical columns in a database.
    Same as impact coding but add a smoothing function instead pof regroup rare modalities:
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    If 2 parameters are given, function (4) of the paper is implemented, else, it's function (8)


    Target variable has to be a number (int or float)

    Parameters
    ----------
    table_in_train : DataFrame
        The train database ; it will be used to compute some values & be encoded

    table_in_test : DataFrame
        The test database ; it will use values from the train database, and will
        be encoded.

    cat_cols : list of Strings
        The columns to encode

    weight : String
        The name of the weight column

    target : String
        The name of the target column

    min_sample : float
        minimum sample size to take category average into account. If lower
        than 1, a % of the train database ; if higher, an absolute number.
        Default = 10%

    smoothing_1 : float
        parameter called k in (4), m in (8) [similar role] in https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
        Smoothing effect to balance categorical average vs prior.
        For now, only a 1 parameter function is coded.
        The parameter determines half of the minimal sample size for which we completely trust the estimate based on
        the sample in the train[var == modality].
        (ex: if you know that modalities that have less than 4% exposure have "too big" error bars,
        smoothing should be around 4%. It will mean that modality with 4% of exposure will have as value:
        0.5*impact_coded_value + 0.5*prior_value)
        If lower than 1, a % of the weights from train database ; if higher, an absolute number.
        Default = 5%

    smoothing_2 : float
        parameter called f in (4), https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
        not tested but number around 0.05 should be a good order of magnitude. (try to play with understand_smoothing_params)
        try to choose f bounded | lambda(0) almost null. (not much bigger than k/2)

    remove_original_cols : Boolean
        Should the original columns be removed after process ?

    verbose : integer
        Should it print comment (1 = few, 2 = a lot) or not (0)

    Returns
    -------

    table_in_train : DataFrame
        The encoded train database ; if inplace == False

    table_in_test : DataFrame
        The encoded train database ; if inplace == False

    impact_cols : List
        The list of the columns created

    """

    table_in_train = table_in_train.copy(deep=True)
    table_in_test = table_in_test.copy(deep=True)

    if table_in_test is None:  # in case we just want to work on a train DB, we create an empty test
        table_in_test = table_in_train[:0].copy(deep=True)
    assert set(cat_cols) < set(table_in_train.columns) and \
           set(cat_cols) < set(table_in_test.columns), \
        "!! columns of train and test should be the same !!"

    # If no weights, add a columns of ones.
    delete_weight = False
    if weight is None:
        weight, delete_weight = 'w', True
        table_in_train = table_in_train.assign(w=np.ones(len(table_in_train)))

    weight_sum = float(table_in_train[weight].sum(axis=0))

    # SMOOTH:
    # smoothing can either be a number of observation or a % of the rows....
    if smoothing_1:
        if smoothing_1 < 1:
            smoothing_1 *= weight_sum

    if smoothing_2:
        if smoothing_2 < 1:
            smoothing_2 *= weight_sum



    impact_cols, count_cols = [], []  # This vector will contain the name of the columns created

    # prior = (table_in_train[target]*table_in_train[weight]).sum()/table_in_train[weight].sum()

    prior, prior_var = weighted_avg_and_var(values=table_in_train[target], weights=table_in_train[weight])


    for colI in cat_cols:
        if verbose >= 1:
            print "impact coding column " + str(colI)
        # First, fill NA with 'OTHER'
        table_in_train[colI] = table_in_train[colI].fillna('OTHER', inplace=False)
        # We will add two columns: one with the average of the target, one with the absolute number of occurrences.
        average_target_i_name, count_col_i_name = colI + "__avg__" + target, colI + "__count"

        # Build a table with the variables of interest alone.
        sub_table_colI = table_in_train[[colI, weight, target]]
        sub_table_colI[count_col_i_name] = 1

        # Build the aggregated table:
        to_aggregate = table_in_train[[target, weight, colI]]

        average_table = pd.pivot_table(to_aggregate.assign(Y_target=lambda t: t[target] * t[weight],
                                                           Y_count=lambda t: t[weight]),
                                       index=[colI],
                                       values=['Y_count', 'Y_target'],
                                       aggfunc={'Y_count': np.sum, 'Y_target': np.sum})
        average_table = average_table.assign(Y_target=lambda t: t["Y_target"] / t['Y_count'])

        # Now add the variance for each modality:
        to_aggregate = pd.merge(to_aggregate, average_table, how="left", left_on=colI, right_index=True)
        av_table_2 = pd.pivot_table(to_aggregate.assign(Y_var=lambda t: t[weight] * (t[target]-t['Y_target'])**2,
                                                        W_2=lambda t: t[weight]**2),
                                    index=colI, values=['Y_var', 'W_2'], aggfunc={'Y_var': np.sum, 'W_2': np.sum})

        average_table = pd.merge(average_table, av_table_2, left_index=True, right_index=True)
        # unbiased variance:
        average_table = average_table.assign(Y_var=lambda t: (t['Y_var'] / (t['Y_count']-(t['W_2']/t['Y_count']))))


        #average_table = average_table.assign(Y_target=lambda t: t["Y_target"] / t['Y_count'])


        if smoothing_2: # if 2 parameters are given, use function (4) of paper
            p = lambda x, k, f: 1 / (1 + np.exp(-(x['Y_count'] - k) / f))
            average_table = average_table.assign(Y_target=lambda t: t["Y_target"] * p(t, smoothing_1, smoothing_2) +
                                                                    (1-p(t, smoothing_1, smoothing_2)) * prior)
        elif smoothing_1:   # if 1 parameter is given, use function (8) of paper
            p = lambda x, m: x['Y_count'] / (m * np.ones(len(average_table)) + x['Y_count'])
            average_table = average_table.assign(Y_target=lambda t: t["Y_target"] * p(t, smoothing_1) +
                                                                    (1-p(t, smoothing_1)) * prior)
        else:  # if 0 param given, use function (8) of paper with m = sigma^2/tau^2:
            p = lambda x: x['Y_count'] / ((x['Y_var']) / (prior_var*np.ones(len(average_table))) + x['Y_count'])
            average_table = average_table.assign(Y_target=lambda t: t["Y_target"] * p(t) + (1-p(t)) * prior)

        average_table.rename(columns={'Y_target': average_target_i_name, 'Y_count': count_col_i_name}, inplace=True)
        average_table.drop('Y_var', inplace=True, axis=1)

        if verbose == 2:
            print average_table

        # We can merge the average target values back in the train and test DBs

        table_in_train = pd.merge(table_in_train, average_table, how="left", left_on=colI, right_index=True)
        table_in_test = pd.merge(table_in_test, average_table, how="left", left_on=colI, right_index=True)

        # If some modalities are in the test DBs but not in the aggregated DB,they will appear as NA...
        # This appends in the rares modalities that are only in the test DB. let's put prior proba on it
        table_in_test[average_target_i_name].fillna(prior, inplace=True)
        # For count, just put -1.
        table_in_test[count_col_i_name].fillna(-1, inplace=True)


        # We add the new column to the list
        impact_cols = impact_cols + [average_target_i_name]
        count_cols = count_cols + [count_col_i_name]
        # Finally, we remove the original column from our database
        if remove_original_cols:
            table_in_train.drop(colI, inplace=True, axis=1)
            table_in_test.drop(colI, inplace=True, axis=1)
        if not add_count_col:
            table_in_train.drop(count_col_i_name, inplace=True, axis=1)
            table_in_test.drop(count_col_i_name, inplace=True, axis=1)


    if delete_weight:  # if no weights in the beginning
        table_in_train.drop(weight, inplace=True, axis=1)


    return table_in_train, table_in_test, impact_cols


def get_cols_types(database, verbose=True):
    """
    :param database: pandas.dataFrame
    The database we want the information on columns from
    :param verbose: bool
    Indicates whether to display verbose
    :return:
    list, list
    List of numerical columns and list of categorical columns
    """

    def get_numerical_columns(df):
        """ Give list of names of numerical columns from dataframe"""
        return list(df.select_dtypes(include=[np.number, 'bool']).columns.values)


    def get_categorical_columns(df):
        """ Give list of names of categorical columns from dataframe"""
        return list(df.select_dtypes(include=['category', 'object']).columns.values)

    num_cols = get_numerical_columns(database)

    cat_cols = get_categorical_columns(database)
    if verbose:
        print 'Number of numerical columns: {}, number of categorical columns: {}, sum: {}'.format(len(num_cols), len(cat_cols), len(num_cols + cat_cols))
        print 'Number of cols database: {}'.format(database.shape[1])
    # Check if nothing has been forgotten
    cols_left_out = set(list(database.columns)) - set(num_cols + cat_cols)
    if cols_left_out != set():
        print "WARNING ! Some columns couldn't be define as numeric nor categorical: {}".format(cols_left_out)
        print "Please check the format and modify the function."
    else:
        print "No problem encountered: all columns were taken into account"

    return num_cols, cat_cols



def create_feature_map(features, path='Results\mapxgb.fmap'):
    """
    Creates and save the file feature map used in xgboost
    Parameters
    ----------
    features: list of str
        names off the fetures
    path: str
        where to save the txt file

    Returns
    -------

    """
    outfile = open(path, 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i += 1
    print 'Feature map saved as: ' + path


def var_importance_xgb(xgb_model, prefix, db_columns, n_var=50, outfile=None, path_save_fmap="Results\\",
                       fmap_name='mapxgb'):
    """

    Parameters
    ----------
    xgb_model: xgboost.train object
        model from which to plot the variable importance
    prefix: str
        Prefix title of the plot saved
    db_columns: list of str
        names of variables /!\ Has to be in the order of the xgb matrix used for training xgb_model
    n_var: int
        Number of variables displayed
    outfile: open file object
        file object to write results (open('workfile', 'w'))
    path_fmap: str
        Path where to write the fmap file.

    Returns
    -------

    """
    path_fmap = path_save_fmap + fmap_name + '.fmap'
    create_feature_map(db_columns, path_fmap)
    importance = xgb_model.get_score(fmap=path_fmap, importance_type='gain')

    importance = sorted(importance.items(), key=itemgetter(1))
    df2 = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df2['fscore'] = df2['fscore'] / df2['fscore'].sum()
    if outfile:
        outfile.write('\n--------------------FEATURE IMPORTANCE------------------\n')
        outfile.write(str(df2))
        outfile.write('\n--------------------------------------------------------\n')

    df2.index = range(0, len(df2.index))
    print df2
    df3 = df2.ix[range((len(df2.index) - n_var + 1), len(df2.index)), ]
    df3.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.gcf().savefig(path_save_fmap + prefix + 'feature_importance.png')


def plot_train_vs_eval(evals_result, prefix, num_round, metric='logloss', path_plot_save='Results\\'):
    """
    Plot the eval_metric on training and validation set for each XGBoost iteration.
    Evals results names have to be 'Train' and 'Val'
    Parameters
    ----------
    evals_result: dict
        watchlist = [(xg_train, 'Train'), (xg_test, 'Val')]
    prefix: str
        prefix of the title of the saved plot
    num_round: int
        Number of gbm iterations (shall be deleted in further versions)
    metric: str
        metric to display (has to be computed by xgboost...)

    Returns
    -------

    """
    plt.figure(figsize=(4, 3))
    plt.plot(range(0, num_round), np.array(evals_result['Train'][metric]).astype(np.float))
    plt.plot(range(0, num_round), np.array(evals_result['Val'][metric]).astype(np.float))
    # np.array(test_b['Train']).astype(np.float)

    y_min = np.array(evals_result['Val'][metric]).astype(np.float).min()

    plt.ylim(y_min, y_min*1.02)
    plt.xlim(0, num_round)
    plt.legend(labels=['Train', 'Val'])
    plt.gcf().savefig(path_plot_save + prefix + '_train_vs_val.png')
    return


def plot_gridsearch(df, col_x, col_y, col_z):
    """

    Parameters
    ----------
    Plot into a 3D graph two parameters against a third.
    df: pandas DataFram
    Dataframe that contains at least three columns that you want to plot.
    col_x: str
    col_y: str
    col_z: str

    Returns
    -------

    """
    fig = plt.figure()
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    ax = fig.gca(projection='3d')
    X, Y, Z = df[col_x], df[col_y], df[col_z]
    ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.2)  # , rstride=8, cstride=8, alpha=0.3)
    ax.set_xlabel(col_x)
    # ax.set_xlim(0, 0.001)
    ax.set_ylabel(col_y)
    ax.set_zlabel(col_z)
    plt.show()
    fig.savefig('Results/gsearch_' + col_x + '_' + col_y + '_' + col_z + '.png', bbox_inches='tight')


def find_categorical_columns(database):
    """
    Find categorical column of a database.
    Parameters
    ----------
    database: pandas DataFrame
    Returns
    -------
    Tuple of str lists: categorical columns and numeric ones.
    """
    type_table = database.dtypes
    numeric_columns, categorical_columns = [], []
    for idx, column in enumerate(database.columns):
        if (type_table[idx] == 'int64') or (type_table[idx] == 'float64'):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    return categorical_columns, numeric_columns


def possible_features_to_numeric(database, rare_freq=1 / 250., fill_rare=-999):
    """
    Done by Aurelien
    :param database: DataFrame to prepare
    :param fill_rare: float or int
        replace rare modalities
    :param rare_freq: float
        threshold to consider a as "rare"
    :return: Dataframe Df with columns checked as follows:
            for each column, if there are non numeric instances inferior to 1/250% of the instances,
             put them to -999.
             Then check if variables can be put to numeric. If yes, do it
             (useful for age for example)
    """
    if rare_freq > 1 or rare_freq < 0:
        raise AttributeError("rare_freq has to be in [0,1]")

    l = len(database)
    t = 0
    can_be_numeric_array = []
    for c in database.columns:
        can_be_numeric = True
        if database[c].dtypes == 'object':
            cnt = Counter(database[c])
            for item in cnt:
                if isinstance(item, basestring):
                    if unicode(item.encode("utf8"), 'utf8').isnumeric():
                        database.loc[database[c] == item, c] = float(item)
                    else:
                        if cnt[item] < l * rare_freq:
                            database.loc[database[c] == item, c] = fill_rare
                        else:
                            can_be_numeric = False
            t += 1
        if can_be_numeric:
            can_be_numeric_array.append(c)

    print can_be_numeric_array
    print "processed ", t, " features out of ", len(database.columns)
    database[can_be_numeric_array] = database[can_be_numeric_array].convert_objects(convert_numeric=True)

    return database


def evaluate_objective_function(db_training, target_name, weight_name, num_rounds, shuffle_split, n_xval,
                                categorical_columns, objective, eta, max_depth, min_child_weight, gamma=0,
                                lambda_=0, colsample_bytree=1, subsample=1, nthread=4, eval_metric='logloss',
                                measure='pseudoR2'):
    """
    For one given set of parameters, evaluate thanks to a "n_xval"-fold cross validation the gini
    of an xgboost model with such parameters on "db_training" dataset.

    Parameters
    ----------
    measure: str
        "pseudoR2" or "gini".
        "pseudoR2" only works with 'reg:gamma', 'count:poisson' and 'reg:linear' for now.
    db_training: pandas DataFrame
        Database on which the gridsearch will be performed
    target_name: str
        target name in db_training
    weight_name: str
        weight name in db_training
    categorical_columns: list of str
        list of the names of the columns being categorical
    n_xval: int
        number of cross validations
    shuffle_split: sklearn.model_selection.ShuffleSplit.split  (scikit version >=0.18.1)
        Split for X-validation. Please refer to scikit-learn help for more information

    Please refer to XGBoost help for the following:
    num_rounds
    objective
    eta
    max_depth
    min_child_weight
    gamma
    lambda_
    colsample_bytree
    subsample
    nthread
    eval_metric

    Returns
    -------
    gini_: float in [0,1]
        Average of the gini on the trains of the Xval
    n_opt
    """

    params = {'objective': objective, 'eval_metric': eval_metric, 'eta': eta, 'max_depth': int(round(max_depth)),
              'subsample': subsample, 'colsample_bytree': colsample_bytree, 'min_child_weight': int(round(min_child_weight)),
              'gamma': gamma, 'lambda': lambda_, 'nthread': nthread}

    # set parameters for xgb
    param_list = list(params.items())

    # Initialize KPIs results
    gini_train, eval_g_train = 0, np.zeros(num_rounds)
    score, eval_g_test = 0, np.zeros(num_rounds)
    evals_result = {}
    gini = 0

    # Cross validation:
    k = 0
    for train_idx, test_idx in shuffle_split:
        k += 1

        # print progress
        print('\n__________ \n X-val %d / %d ______________' % (k, n_xval))
        print('gamma = %f ; lambda = %f ; Depth max = %d ; Min child = %d ; subsample = %f ; colsample_bytree = %f'
              % (params['gamma'], params['lambda'], params['max_depth'], params['min_child_weight'],
                 params['subsample'], params['colsample_bytree']))

        #  Split into test/train inside X-val
        train, test = db_training.iloc[train_idx], db_training.iloc[test_idx]
        #  code categorical columns
        # train, test, impact_cols = create_impact_coding(train, test, categorical_columns, target_name,
        #                                                 weight=weight_name, regroup_rare_modalities_threshold=0.1,
        #                                                 remove_original_cols=True, verbose=0)

        train, test, impact_cols = create_smooth_impact_coding(train, test, categorical_columns, target_name,
                                                               weight=weight_name, remove_original_cols=True, verbose=0)


        # Prepare into X,y and w for XGBoost
        x_train = train.drop([target_name, weight_name], inplace=False, axis=1)
        x_test = test.drop([target_name, weight_name], inplace=False, axis=1)
        y_train, y_test = train[target_name], test[target_name]
        w_train, w_test = train[weight_name], test[weight_name]

        # XGBoost
        xg_train = xgb.DMatrix(x_train.values, label=y_train.values, weight=w_train.values)
        xg_test = xgb.DMatrix(x_test.values, label=y_test.values, weight=w_test.values)
        watchlist = [(xg_train, 'Train_Xval'), (xg_test, 'Test_Xval')]
        plst = params.items()
        gbm_model = xgb.train(plst, xg_train, num_rounds, watchlist, evals_result=evals_result, verbose_eval=50)

        y_prediction = gbm_model.predict(xg_test)
        # KPIs
        eval_g_train += np.array(evals_result['Train_Xval'].values()[0]).astype(float)
        eval_g_test += np.array(evals_result['Test_Xval'].values()[0]).astype(float)
        gini += gini_emblem_fast(y_test, y_prediction, weights=w_test, normalize_gini=False)
        if measure == 'gini':
            score += gini
        elif measure == 'pseudoR2':
            if objective == 'reg:gamma':
                score += gamma_pseudo_r2(np.array(y_test), np.array(y_prediction), weight=np.array(w_test))
            if objective == 'count:poisson':
                score += poisson_pseudo_r2(np.array(y_test), np.array(y_prediction), weight=np.array(w_test))
            if objective == 'reg:linear':
                score += gaussian_pseudo_r2(np.array(y_test), np.array(y_prediction), weight=np.array(w_test))
        else:
            raise ValueError('measure has to be "gini" or "pseudoR2". If pseudoR2, please read function description')
    # Average the result


    eval_g_train, eval_g_test = eval_g_train / float(n_xval), eval_g_test / float(n_xval)
    score /= float(n_xval)
    gini /= float(n_xval)
    n_opt = np.argmin(eval_g_test)  # iteration where the error is minimal

    return score, n_opt, gini,  eval_g_train, eval_g_test


def evaluate_objective_function3(db_training, target_name, num_rounds, shuffle_split, n_xval,
                                categorical_columns, objective, eta, max_depth, min_child_weight, gamma=0,
                                lambda_=0, colsample_bytree=1, subsample=1, nthread=4, eval_metric='logloss',
                                measure='pseudoR2', weight_name=None, offset=None):
    """
    For one given set of parameters, evaluate thanks to a "n_xval"-fold cross validation the gini
    of an xgboost model with such parameters on "db_training" dataset.

    Parameters
    ----------
    measure: str
        "pseudoR2" or "gini".
        "pseudoR2" only works with 'reg:gamma', 'count:poisson' and 'reg:linear' for now.
    db_training: pandas DataFrame
        Database on which the gridsearch will be performed
    target_name: str
        target name in db_training
    weight_name: str
        weight name in db_training
    categorical_columns: list of str
        list of the names of the columns being categorical
    n_xval: int
        number of cross validations
    shuffle_split: sklearn.model_selection.ShuffleSplit.split  (scikit version >=0.18.1)
        Split for X-validation. Please refer to scikit-learn help for more information
    offset: str
        The offset has to be a column in the dataframe, and to precise the name here.
         This can be used to specify a prediction value of existing model to be base_margin
        However, remember margin is needed, instead of transformed prediction
        e.g. for logistic regression: need to put in value before logistic transformation
        see also in XGBoost git: example/demo.py

    Please refer to XGBoost help for the following:
    num_rounds
    objective
    eta
    max_depth
    min_child_weight
    gamma
    lambda_
    colsample_bytree
    subsample
    nthread
    eval_metric

    Returns
    -------
    gini_: float in [0,1]
        Average of the gini on the trains of the Xval
    n_opt
    """

    params = {'objective': objective, 'eval_metric': eval_metric, 'eta': eta, 'max_depth': int(round(max_depth)),
              'subsample': subsample, 'colsample_bytree': colsample_bytree, 'min_child_weight': int(round(min_child_weight)),
              'gamma': gamma, 'lambda': lambda_, 'nthread': nthread}

    # set parameters for xgb
    param_list = list(params.items())

    # Initialize KPIs results
    gini_train, eval_g_train = 0, np.zeros(num_rounds)
    score, eval_g_test = 0, np.zeros(num_rounds)
    evals_result = {}
    gini = 0

    # Cross validation:
    k = 0
    for train_idx, test_idx in shuffle_split:
        k += 1

        # print progress
        print('\n__________ \n X-val %d / %d ______________' % (k, n_xval))
        print('gamma = %f ; lambda = %f ; Depth max = %d ; Min child = %d ; subsample = %f ; colsample_bytree = %f'
              % (params['gamma'], params['lambda'], params['max_depth'], params['min_child_weight'],
                 params['subsample'], params['colsample_bytree']))

        #  Split into test/train inside X-val
        train, test = db_training.iloc[train_idx], db_training.iloc[test_idx]
        # if not None, define offset
        if offset:
            offset_train, offset_test = train[offset].values, test[offset].values
            if (offset != weight_name):
                train.drop([offset], inplace=True, axis=1)
                test.drop([offset], inplace=True, axis=1)
        #  Impact coding on categorical columns
        train, test, impact_cols = create_impact_coding(train, test, categorical_columns, target_name,
                                                        weight=weight_name, regroup_rare_modalities_threshold=0.1,
                                                        remove_original_cols=True, verbose=0)

        # Prepare into X,y and w for XGBoost
        if weight_name is None:
        	weight_name = 'w'
        	train[weight_name], test[weight_name] = np.ones(len(train)), np.ones(len(test))

		x_train = train.drop([target_name, weight_name], inplace=False, axis=1)
		x_test = test.drop([target_name, weight_name], inplace=False, axis=1)
		y_train, y_test = train[target_name], test[target_name]
		w_train, w_test = train[weight_name], test[weight_name]

        # XGBoost
        xg_train = xgb.DMatrix(x_train.values, label=y_train.values, weight=w_train.values)
        xg_test = xgb.DMatrix(x_test.values, label=y_test.values, weight=w_test.values)
        if offset:
            xg_train.set_base_margin(np.log(offset_train))
            xg_test.set_base_margin(np.log(offset_test))


        watchlist = [(xg_train, 'Train_Xval'), (xg_test, 'Test_Xval')]
        plst = params.items()
        gbm_model = xgb.train(plst, xg_train, num_rounds, watchlist, evals_result=evals_result, verbose_eval=50)
        # predict
        y_prediction = gbm_model.predict(xg_test)
        # KPIs
        eval_g_train += np.array(evals_result['Train_Xval'].values()[0]).astype(float)
        eval_g_test += np.array(evals_result['Test_Xval'].values()[0]).astype(float)
        gini += gini_emblem_fast(y_test, y_prediction, weights=w_test, normalize_gini=False)
        if measure == 'gini':
            score += gini
        elif measure == 'pseudoR2':
            if objective == 'reg:gamma':
                score += gamma_pseudo_r2(np.array(y_test), np.array(y_prediction), weight=np.array(w_test))
            if objective == 'count:poisson':
                score += poisson_pseudo_r2(np.array(y_test), np.array(y_prediction), weight=np.array(w_test))
            if objective == 'reg:linear':
                score += gaussian_pseudo_r2(np.array(y_test), np.array(y_prediction), weight=np.array(w_test))
        else:
            raise ValueError('measure has to be "gini" or "pseudoR2". If pseudoR2, please read function description')
    # Average the result


    eval_g_train, eval_g_test = eval_g_train / float(n_xval), eval_g_test / float(n_xval)
    score /= float(n_xval)
    gini /= float(n_xval)
    n_opt = np.argmin(eval_g_test)  # iteration where the error is minimal

    return score, n_opt, gini,  eval_g_train, eval_g_test


def bayes_gridsearch(db_training, target_name, categorical_columns, weight_name, dict_bounds, num_rounds,
                     kappa=3, n_Xval=5, n_points=10, nthread=4, eval_metric='logloss', objective='count:poisson',
                     outfile=None, measure="pseudoR2"):
    """
    Bayesian optimized gridsearch. For more information:
    https://github.com/fmfn/BayesianOptimization/blob/master/examples/visualization.ipynb
    For now, only optimizes the Gini.
    Can now do the process optimizing Gini (measure="gini") or Deviance (measure="pseudoR2")

    Parameters
    ----------


    kappa: float
        Bayesian parameter - the bigger, the bolder the algorithm.
    db_training: pandas DataFrame
    target_name: str
    categorical_columns: list of str
    weight_name: str
    dict_bounds: dictionary
        Dictionary defining a bounds sup and inf for each parameter: {'eta': [0.0001,0.1],...
        To fix one parameter: 'lambda': [0,0]
        Be careful of integer parameters: ( I don't know what it's doing... see #todo)
    num_rounds: int
    n_Xval: int
    n_points: int
        Number of points to compute for the gridsearch.
    nthread: int
    eval_metric: str
    objective: str
    outfile: opened file object
        file object to write results (open('workfile', 'w'))
    measure:
        The measure we want to optimize by gridsearching. "gini" or "pseudoR2"

    Returns
    -------
    pandas DataFrame describing gridsearch results.
    """

    # TODO: search what to do for integers parameters
    # (think it just put floats and xgboost rounds it? So be careful, bayesian opti could think of values on points
    # that are in fact other points...
    global list_of_optimal_n, list_of_scores, list_of_gini  # to improve
    list_of_optimal_n, list_of_gini = [], []


    def target(eta, max_depth, min_child_weight, gamma, lambda_, colsample_bytree, subsample):
        """
        target for BayesianOptimisation, have to return only the value of the point where the function is being assessed.
        Bayesian optimization search for a maximum, so return -s if optimization = 'minimize'

        """
        kf = KFold(n_splits=n_Xval, random_state=45, shuffle=True)

        s, n, g = evaluate_objective_function(db_training, target_name, weight_name, num_rounds, kf.split(db_training),
                                           n_Xval, categorical_columns, objective, eta, max_depth, min_child_weight,
                                           gamma, lambda_, colsample_bytree, subsample, nthread, eval_metric,
                                           measure=measure)[0:3]

        list_of_optimal_n.append(n)  # to get it outside bayesian optimization
        list_of_gini.append(g)
        return s

    # Initialize : first point: mean of the bounds.
    init_point = pd.DataFrame(pd.DataFrame(dict_bounds).mean()).transpose()
    # Compute the value for that first point
    init_score = target(init_point['eta'][0], init_point['max_depth'][0], init_point['min_child_weight'][0],
                        init_point['gamma'][0], init_point['lambda_'][0], init_point['colsample_bytree'][0],
                        init_point['subsample'][0])

    init_point = init_point.assign(target=init_score)
    # Build the Bayesian Optimizer object
    bo = BayesianOptimization(target, dict_bounds)
    # Initialize with the point just computed.
    bo.initialize_df(init_point)
    # Compute the other points.
    bo.maximize(init_points=0, n_iter=n_points, acq='ucb', kappa=kappa)

    # Finally, aggregate results in DataFrame.
    init_point.rename(columns={'target': 'values'}, inplace=True)
    df_ = pd.DataFrame(bo.res['all']['params']).assign(values=bo.res['all']['values'])
    df_ = init_point.append(df_, ignore_index=True)
    df_ = df_.assign(num_rounds=list_of_optimal_n)
    df_ = df_.assign(gini_test=list_of_gini)
    df_.sort_values(by='values', axis=0, ascending=False, inplace=True)
    df_.rename(columns={'lambda_': 'lambda'}, inplace=True)
    df_ = df_.assign(eval_metric=np.repeat(eval_metric, len(df_)))

    if outfile:
        outfile.write(str(df_))

    return df_, bo


def bayes_gridsearch2(db_training, target_name, categorical_columns, dict_bounds, num_rounds,
                     kappa=3, n_Xval=5, n_points=10, nthread=4, eval_metric='logloss', objective='count:poisson',
                     outfile=None, measure="pseudoR2", weight_name=None, offset=None):
    """
    Bayesian optimized gridsearch. For more information:
    https://github.com/fmfn/BayesianOptimization/blob/master/examples/visualization.ipynb
    For now, only optimizes the Gini.
    Can now do the process optimizing Gini (measure="gini") or Deviance (measure="pseudoR2")

    Parameters
    ----------


    kappa: float
        Bayesian parameter - the bigger, the bolder the algorithm.
    db_training: pandas DataFrame
    target_name: str
    categorical_columns: list of str
    weight_name: str
    offset: str
    dict_bounds: dictionary
        Dictionary defining a bounds sup and inf for each parameter: {'eta': [0.0001,0.1],...
        To fix one parameter: 'lambda': [0,0]
        Be careful of integer parameters: ( I don't know what it's doing... see #todo)
    num_rounds: int
    n_Xval: int
    n_points: int
        Number of points to compute for the gridsearch.
    nthread: int
    eval_metric: str
    objective: str
    outfile: opened file object
        file object to write results (open('workfile', 'w'))
    measure:
        The measure we want to optimize by gridsearching. "gini" or "pseudoR2"

    Returns
    -------
    pandas DataFrame describing gridsearch results.
    """

    # TODO: search what to do for integers parameters
    # (think it just put floats and xgboost rounds it? So be careful, bayesian opti could think of values on points
    # that are in fact other points...
    global list_of_optimal_n, list_of_scores, list_of_gini  # to improve
    list_of_optimal_n, list_of_gini = [], []


    def target(eta, max_depth, min_child_weight, gamma, lambda_, colsample_bytree, subsample):
        """
        target for BayesianOptimisation, have to return only the value of the point where the function is being assessed.
        Bayesian optimization search for a maximum, so return -s if optimization = 'minimize'

        """
        kf = KFold(n_splits=n_Xval, random_state=45, shuffle=True)

        s, n, g = evaluate_objective_function3(db_training, target_name, num_rounds, kf.split(db_training),
                                           n_Xval, categorical_columns, objective, eta, max_depth, min_child_weight,
                                           gamma, lambda_, colsample_bytree, subsample, nthread, eval_metric,
                                           measure, weight_name, offset)[0:3]

        list_of_optimal_n.append(n)  # to get it outside bayesian optimization
        list_of_gini.append(g)
        return s

    # Initialize : first point: mean of the bounds.
    init_point = pd.DataFrame(pd.DataFrame(dict_bounds).mean()).transpose()
    # Compute the value for that first point
    init_score = target(init_point['eta'][0], init_point['max_depth'][0], init_point['min_child_weight'][0],
                        init_point['gamma'][0], init_point['lambda_'][0], init_point['colsample_bytree'][0],
                        init_point['subsample'][0])

    init_point = init_point.assign(target=init_score)
    # Build the Bayesian Optimizer object
    bo = BayesianOptimization(target, dict_bounds)
    # Initialize with the point just computed.
    bo.initialize_df(init_point)
    # Compute the other points.
    bo.maximize(init_points=0, n_iter=n_points, acq='ucb', kappa=kappa)

    # Finally, aggregate results in DataFrame.
    init_point.rename(columns={'target': 'values'}, inplace=True)
    df_ = pd.DataFrame(bo.res['all']['params']).assign(values=bo.res['all']['values'])
    df_ = init_point.append(df_, ignore_index=True)
    df_ = df_.assign(num_rounds=list_of_optimal_n)
    df_ = df_.assign(gini_test=list_of_gini)
    df_.sort_values(by='values', axis=0, ascending=False, inplace=True)
    df_.rename(columns={'lambda_': 'lambda'}, inplace=True)
    df_ = df_.assign(eval_metric=np.repeat(eval_metric, len(df_)))

    if outfile:
        outfile.write(str(df_))

    return df_, bo


def bayes_predict_best_point(bo, resolution=10):
    """
    Uses the bayesian optimizer, the points choosen and the kernel for estimating the whole solution space.
    you can choose the number of points/parameters thanks to 'resolution'.

    Parameters
    ----------
    bo: bayesian optimisation object
    resolution: int
        number of points max for a parameter. Please remember that in case of xgboost, there are 6 parameters.
        Choosing a too big number can slow down the thing if there are bounds for each parameters.
        for example for resolution = 10, it has to compute 10^6 points
        If you have bounds like (0,0), you'll delete one dimension. (10^5)

    Returns
    -------
    pandas DataFrame containing the evaluated points and the points the 'sure' points sorted following the score.
    """
    dict_linspace = {}
    for key, bounds in zip(bo.keys, bo.bounds):
        dict_linspace[key] = np.linspace(bounds[0], bounds[1], resolution)

    xnp = np.array(list((itertools.product(np.unique(dict_linspace[bo.keys[0]]),
                                           np.unique(dict_linspace[bo.keys[1]]),
                                           np.unique(dict_linspace[bo.keys[2]]),
                                           np.unique(dict_linspace[bo.keys[3]]),
                                           np.unique(dict_linspace[bo.keys[4]]),
                                           np.unique(dict_linspace[bo.keys[5]]),
                                           np.unique(dict_linspace[bo.keys[6]])))))

    #  add the points really evaluated
    xnp = np.vstack([xnp, bo.X])

    y_mean, y_std = bo.gp.predict(xnp, return_std=True)
    df = pd.DataFrame(xnp, columns=bo.keys).assign(score_estimation=y_mean, score_std=y_std)

    df.sort_values(by='score_estimation', axis=0, ascending=False, inplace=True)

    return df


def normal_gridsearch(db_training, target_name, weight_name, categorical_columns, n_Xval, eta, num_rounds, lambda_,
                      gamma, max_depth, min_child_weight, eval_metric, objective, colsample_bytree, subsample,
                      outfile=None,
                      nthread=4):
    """

    Parameters
    ----------
    db_training: pandas DataFrame
        Database on which the gridsearch will be performed
    target_name: str
        target name in db_training
    weight_name: str
        weight name in db_training
    categorical_columns: list of str
        list of the names of the columns being categorical
    n_Xval: int
        number of cross validations

    Please check XGB documentation for the followings
    eta: list
    shrink parameter:list
    num_rounds: int
    lambda_: list
    gamma: list
    max_depth: list
    min_child_weight: list
    eval_metric: list
    objective: list
    colsample_bytree: list
    subsample: list
    outfile: open file object
        file object to write results (open('workfile', 'w'))
    nthread: int
        number of threads used for computing trees.
    Returns
    -------
    Pandas DataFrame containing all the results of the GridSearch

    """

    exp_plan = pd.DataFrame(list(
        itertools.product(eta, num_rounds, lambda_, gamma, max_depth, min_child_weight, colsample_bytree, subsample,
                          eval_metric)),
        columns=['eta', 'num_rounds', 'lambda', 'gamma', 'max_depth', 'min_child_weight', 'colsample_bytree',
                 'subsample',
                 'eval_metric'])

    # for Scikit < 0.18.1:
    # sp = cross_validation.ShuffleSplit(len(db_training), n_iter=n_Xval, test_size=1 / float(n_Xval), random_state=45)


    result = []
    print 'Starting Gridsearch.\n %d combinations to test, with %d X-validation.\nTotal: %d XGBoosts' % (
        len(exp_plan), n_Xval, n_Xval * len(exp_plan))
    print exp_plan

    for i in range(exp_plan.shape[0]):  # for each combination of parameters:
        sp = KFold(n_splits=n_Xval, random_state=45, shuffle=True).split(db_training)
        eval_ = evaluate_objective_function(db_training=db_training,
                                            target_name=target_name,
                                            weight_name=weight_name,
                                            num_rounds=exp_plan['num_rounds'].iloc[i],
                                            shuffle_split=sp,
                                            n_xval=n_Xval,
                                            categorical_columns=categorical_columns,
                                            objective=objective,
                                            eta=exp_plan['eta'].iloc[i],
                                            max_depth=exp_plan['max_depth'].iloc[i],
                                            min_child_weight=exp_plan['min_child_weight'].iloc[i],
                                            gamma=exp_plan['gamma'].iloc[i],
                                            lambda_=exp_plan['lambda'].iloc[i],
                                            colsample_bytree=exp_plan['colsample_bytree'].iloc[i],
                                            subsample=exp_plan['subsample'].iloc[i],
                                            nthread=nthread,
                                            eval_metric=exp_plan['eval_metric'].iloc[i])

        gini_, n_opt, eval_g_train, eval_g_test = eval_
        # write results in the txt outfile
        if outfile:
            outfile.write('\n\n_____________________________BEST VALUE_______________________________\n\n')
            outfile.write('gamma: ' + str(exp_plan['gamma'].iloc[i]) + '\nlambda: ' + str(exp_plan['lambda'].iloc[i]))
            outfile.write('\nmax_depth: ' + str(exp_plan['max_depth'].iloc[i]) + '\nMin Child: ' + str(
                exp_plan['min_child_weight'].iloc[i]) + '\nColSample: ' + str(
                exp_plan['colsample_bytree'].iloc[i]) + '\nSubsample:' + str(exp_plan['subsample'].iloc[i]))
            outfile.write('\nBest mean: ' + str(eval_g_test.min()) + '\tattained for num_rounds = ' + str(n_opt))
            outfile.write('\nGini: ' + str(gini_) + '\n\n_________________________________________\n\n')
            # Write results for the function

        result.append(pd.DataFrame(
            [[exp_plan['eta'].iloc[i], exp_plan['num_rounds'].iloc[i], exp_plan['lambda'].iloc[i],
              exp_plan['gamma'].iloc[i],
              exp_plan['max_depth'].iloc[i], exp_plan['min_child_weight'].iloc[i], exp_plan['subsample'].iloc[i],
              exp_plan['colsample_bytree'].iloc[i], exp_plan['eval_metric'].iloc[i],
              eval_g_train.min(), eval_g_test.min(), gini_, n_opt]],
            columns=['eta', 'num_rounds', 'lambda', 'gamma', 'max_depth', 'min_child_weight', 'subsample',
                     'colsample_bytree',
                     'eval_metric', 'score_train', 'score_val', 'Gini Val', 'Optimal Iteration']))

    df_results = pd.concat(result, ignore_index=True)

    return df_results


def predict_on_holdout(train_db, test_db, target_name, weight_name, rdm10_name, categorical_columns, params_line,
                       objective='count:poisson', nthread=10, path_plot_save=None, add_log=True, outfile=None):
    """
    Learn a XGBoost on train_db, apply it to train_db and test_db , concat the 2 and re-sort it as it was.
    Add prediction and log(prediction) to the database.
    /!\ Train and test have to contain rdm10 variable.
    /!\ db_tot will contain the impact coded variables.
    Parameters
    ----------

    train_db: pandas DataFrame
    test_db: pandas DataFrame
    target_name: str
    weight_name: str
    rdm10_name: str
    categorical_columns: list of str
    params_line: pandas DataFrame
        output of gridsearch, the first line will be used as parameters
    objective: str
        XGBoost objective function
    nthread: int
        Number of threads to use for XgBoost
    path_plot_save: str
        don't put the name of the file, just the path where you want it saved. If None, no plot is saved.
    add_log: bool
        Add also the log of the prediction into the database.
    outfile: opened file object
        Write ginis in the specified outfile if not None
    Returns
    -------
    - db_tot: pandas DataFrame
        be careful, the variables are "impact coded"
    - gbm_model: xgboost.train object
    """

    def set_params(table_best_line, objective, nthread):
        """

        Parameters
        ----------
        table_best_line: pandas DataFrame
            DataFrame output of gridsearch. takes the first line, so it has to be sorted.
        objective: str
            xgboost objective function
        nthread: int
            xgboost nthread

        Returns
        -------

        """
        params["objective"] = objective  # 'reg:linear'#'rank:pairwise'#'count:poisson'
        params['eval_metric'] = table_best_line['eval_metric'].iloc[0]
        params["eta"] = table_best_line['eta'].iloc[0]
        params["max_depth"] = int(table_best_line['max_depth'].iloc[0])
        params["subsample"] = table_best_line['subsample'].iloc[0]
        params["colsample_bytree"] = table_best_line['colsample_bytree'].iloc[0]
        params["min_child_weight"] = int(table_best_line['min_child_weight'].iloc[0])
        params["silent"] = 1
        params["nthread"] = int(nthread)
        params["gamma"] = table_best_line['gamma'].iloc[0]
        params["lambda"] = table_best_line['lambda'].iloc[0]
        return params

    # set parameters
    params = {}
    set_params(params_line, objective, nthread)
    plst = list(params.items())
    num_rounds = params_line['num_rounds'].iloc[0]

    # Initialize KPIs results
    evals_result = {}

    # code categorical columns
    train_db, test_db, impact_cols = create_impact_coding(train_db, test_db, categorical_columns, target_name,
                                                          weight=weight_name, regroup_rare_modalities_threshold=0.1,
                                                          remove_original_cols=True, verbose=1)

    # train_db, test_db, impact_cols = create_smooth_impact_coding(train_db, test_db, categorical_columns, target_name,
    #                                                        weight=weight_name, remove_original_cols=True, verbose=0)
    print test_db.shape, train_db.shape


    # db_tot = pd.concat([train_db, test_db]).sort_index()

    # Prepare into X, y and w for XGBoost
    Xtrain = train_db.drop([target_name, weight_name, rdm10_name], inplace=False, axis=1)
    Xtest = test_db.drop([target_name, weight_name, rdm10_name], inplace=False, axis=1)
    ytrain, ytest = train_db[target_name], test_db[target_name]
    wtrain, wtest = train_db[weight_name], test_db[weight_name]

    # XGBoost
    xgtrain = xgb.DMatrix(Xtrain.values, label=ytrain.values, weight=wtrain.values)
    xgtest = xgb.DMatrix(Xtest.values, label=ytest.values, weight=wtest.values)
    watchlist = [(xgtrain, 'Train'), (xgtest, 'Val')]
    gbm_model = xgb.train(plst, xgtrain, num_rounds, watchlist, evals_result=evals_result)
    preds_test = gbm_model.predict(xgtest)
    preds_train = gbm_model.predict(xgtrain)

    # Variable importance
    if path_plot_save:
        var_importance_xgb(gbm_model, '', Xtrain.columns, path_save_fmap=path_plot_save)

    # Gini
    print '____________________________________________________________________________________\nGini Validation set:\n'
    gini_val = gini_emblem_fast(y=ytest, y_pred=preds_test, weights=wtest, normalize_gini=True, verbose=True)
    print '_________________________________________________________________________________________\nGini Train set:\n'
    gini_train = gini_emblem_fast(y=ytrain, y_pred=preds_train, weights=wtrain, normalize_gini=True, verbose=True)
    if outfile:
        outfile.write('Gini Validation set:' + str(gini_val))
        outfile.write('Gini Train set:' + str(gini_train))

    # Lift
    if path_plot_save:
        plot_lift_curve(n_band=50, y_pred=preds_test, y=ytest, weight=wtest, title='lift curve', path_plot_save=path_plot_save)
        # train_vs_eval
        plot_train_vs_eval(evals_result=evals_result, prefix='test', num_round=num_rounds, metric=params['eval_metric'], path_plot_save=path_plot_save)

    db_tot = pd.concat([train_db.assign(gbm_pred=preds_train), test_db.assign(gbm_pred=preds_test)]).sort_index()
    if add_log:
        db_tot = db_tot.assign(gbm_pred_log=np.log(db_tot['gbm_pred']))

    return db_tot, gbm_model



def interaction_importance_xgb(xgb_model, fmap_path, title, saving_path='Results\\'):
    """
    This function is a xgbfir wrapper. Writes an .xlsx file with interaction importance,
    and correct the bug not displaying the variables.

    Parameters
    ----------
    xgb_model: xgboodst.train object
        The model from where is to see interaction importance
    fmap_path: str
        Path of the fmap file (created with create_feature_map() )
    saving_path:str
        File path describing where the excel sheet has to be saved.
    Returns
    -------
    Nothing
    """

    # Create path if it doesn't exists.
    # if not os.path.exists(saving_path):
    #     os.makedirs(saving_path)

    def create_vba_code(fmap_path, saving_path):
        """
        Bug in xgbfir for some versions of xgBoost: the variables names aren't replaced.
        This code writes a VBA codes to correct the problem and save it as text_vba.txt )
        Parameters
        ----------
        fmap_path: str
            path of the fmap file

        Returns
        -------
        Pandas DataFrame containing the real name of the variable and the corresponding name used by xgb
        """
        corresp = pd.read_table(fmap_path, '\t', usecols=[0, 1], header=None, names=['feat', 'name'])
        corresp.feat = corresp.feat.apply(lambda x: 'f' + str(x))

        f = open(saving_path + 'script_vba.txt', 'w')

        script = '''Sub add_bars()
            Application.ScreenUpdating = False
            WS_Count = ActiveWorkbook.Worksheets.Count


            For K = 1 To WS_Count
                Worksheets(K).Activate
                lastRow = ActiveSheet.Cells(ActiveSheet.Rows.Count, "A").End(xlUp).Row
                If Worksheets(K).Name <> "Split Value Histograms" Then
                    For I = 2 To lastRow
                        Value = "|" + ActiveSheet.Cells(I, 1) + "|"
                        ActiveSheet.Cells(I, 1) = Value
                    Next I
                End If
            Next K

        End Sub

        Sub find_and_replace(find, replace)

            WS_Count = ActiveWorkbook.Worksheets.Count

            bar_find = "|" + find + "|"
            bar_replace = "|" + replace + "|"

            For K = 1 To WS_Count
                Worksheets(K).Activate
                a = Cells.replace(What:=bar_find, Replacement:=bar_replace, LookAt:=xlPart)
            Next K

            Worksheets("Split Value Histograms").Activate
            a = Cells.replace(What:=find, Replacement:=replace, LookAt:=xlWhole)
        End Sub

        Sub main()
            Call add_bars()
        '''

        for i in range(len(corresp)):
            script += '\n\t\tCall find_and_replace("' + str(corresp.feat[i]) + '", "' + str(corresp.name[i]) + '")'

        script += '\nApplication.ScreenUpdating = True\nEnd Sub'
        f.write(script)
        f.close()
        return corresp

    corr = create_vba_code(fmap_path, saving_path)

    xgbfir.saveXgbFI(xgb_model, OutputXlsxFile=saving_path + title + '.xlsx')


