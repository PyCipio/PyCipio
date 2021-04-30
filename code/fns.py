## import packages
import pymc3 as pm
import pandas as pd 
import numpy as np 
import seaborn as sns
import arviz as az
import matplotlib.pyplot as plt
import os 
import theano
import random

## make train/test split. 
def train_test(d, idx_col, train_size = .75):
    """create train-test split of pd.DataFrame. 

    Args:
        d (pd.DataFrame): the main data frame. 
        idx_col (str): the col to split by (e.g. Year).
        train_size (float, optional): training partition size. Defaults to .75.

    Returns:
        [tuple]: x_train and y_train to be unpacked.
    """    

    # pretty manual approach
    sort_val = np.sort(np.unique(d[idx_col].values))
    min_val = min(sort_val)
    length = int(round(len(sort_val)*train_size, 0))
    train = d[d[idx_col] <= min_val+length]
    test = d[d[idx_col] > min_val+length]
    return train, test

## get idx for both grouping and time variable. 
def get_idx(d, group_col, time_col, group_name, time_name):
    """get idx grouping column and time-column starting from 0. 

    Args:
        d (pd.DataFrame): Your dataframe
        group_col (str): Name of grouping column (e.g. Country/Company)
        time_col (str): Name of time column (e.g. Year/Minute)
        group_name (str): Name of grouping column (e.g. idx)
        time_name (str): Name of new time column (e.g. time)

    Returns:
        pd.DataFrame: The input pd.DataFrame with two new columns
    """    
    
    d[group_name] = pd.Categorical(d[group_col]).codes
    d[time_name] = pd.Categorical(d[time_col]).codes
    
    return d

## posterior predictive on new data (test data). 
def pp_test(mod_name, trace, dct, params):
    """posterior predictive on unseen data. 

    Args:
        mod_name (pymc3.model.Model): pymc3 model object. 
        trace (arviz.data.inference_data.InferenceData): Model trace (from pm.trace())
        dct (dict): Dictionary with variables to change (e.g. x variable). 
        params (list): Parameters (e.g. alpha, beta, y_pred)

    Returns:
        dict: dictionary with predictive draws for each parameter specified.
    """    
    with mod_name: 
        pm.set_data(dct)
        predictions = pm.sample_posterior_predictive(trace, var_names = params)
        return predictions


## plot predictive (for linear for now). 
def plot_pp(pp, train, test, d, x, y, label_column, 
            lines = True, size = 50, std = 3):
    """plot posterior predictive on unseen data (& the training data).

    Args:
        pp (dct): dictionary of predictions.
        train (pd.DataFrame): the train pd.DataFrame.
        test (pd.DataFrame): the test pd.DataFrame.
        d (pd.DataFrame): the name of your full pd.DataFrame.
        x (str): The name you specified earlier for new x value "x_new". 
        y (str): The name of your y variable.      
        label_column (str): The name of your label column (e.g. "Country"/"Company"). 
        lines (bool, optional): Kruschke style draws. Defaults to True.
        size (int, optional): number of draws. Defaults to 50.
        std (int, optional): standard deviation for shaded area. Defaults to 3.
    """    

    '''
    ### to be done. 
    1. Better plot format in cases of many subplots?
    2. I don't think we need the if statement if we are smart. 
    '''
    
    # go into data and take out labels 
    labels = np.unique(d[label_column].values)
    len_idx = len(labels)
    
    # needed variables (double assignment)
    len_test_rolling = len_test = len(np.unique(test[x].values))
    len_train_rolling = len_train = len(np.unique(train[x].values))
    draws = pp['α'].shape[0]
    y_pred = pp["y_pred"]
    
    # the loop.
    for i in range(len_idx):
        
        ### if only one level 
        if len_idx == 1: 
            y_pred_mean = y_pred.mean(axis=0)
            y_pred_std = y_pred.std(axis=0)
            plt.figure(figsize=(16, 8))
            plt.scatter(train[x].values, train[y].values, c='k', zorder=10, label='Data')
            plt.scatter(test[x].values, test[y].values, zorder = 10, c="red", label='Held-out')
            plt.plot(test[x].values, y_pred_mean, label='Prediction Mean', linewidth = 5, c = "k")
            plt.fill_between(test[x].values, y_pred_mean - std*y_pred_std, y_pred_mean + std*y_pred_std, 
                            alpha=0.2, label='Uncertainty Interval ($\mu\pm3\sigma$)')
        
        ### if multilevel
        if len_idx > 1: 
            y_pred_mean = y_pred[:, len_test_rolling-len_test:len_test_rolling].mean(axis=0)
            y_pred_std = y_pred[:, len_test_rolling-len_test:len_test_rolling].std(axis=0)
            plt.figure(figsize=(16, 8))
            plt.scatter(train[x].values[len_train_rolling-len_train:len_train_rolling], #one further up?
                        train[y].values[len_train_rolling-len_train:len_train_rolling], 
                        c='k', zorder = 10, label='Data')
            plt.scatter(test[x].values[len_test_rolling-len_test:len_test_rolling], 
                        test[y].values[len_test_rolling-len_test:len_test_rolling], 
                        c="red", label='Held-out')
            plt.plot(test[x].values[len_test_rolling-len_test:len_test_rolling], 
                    y_pred_mean, 
                    label='Prediction Mean', linewidth = 5, c = "k")
            plt.fill_between(test[x].values[len_test_rolling-len_test:len_test_rolling], 
                            y_pred_mean - std*y_pred_std, y_pred_mean + std*y_pred_std, 
                            alpha=0.2, label='Uncertainty Interval ($\mu\pm3\sigma$)')
            len_test_rolling += len_test
            len_train_rolling += len_train
            
        ## optionally add lines 
        if lines == True: 

            # should take a pair. (randomize index).
            samples = [random.randrange(draws) for x in range(size)]

            ### if only one level
            if len_idx == 1: 
                alpha = [pp["α"][sample] for sample in samples]
                beta = [pp["β"][sample] for sample in samples]

            ### if multilevel
            if len_idx > 1: 
                alpha = [pp["α"][sample, i] for sample in samples]
                beta = [pp["β"][sample, i] for sample in samples]

            for a, b in zip(alpha, beta):
                y_predictions = a + b * test[x].values
                plt.plot(test[x].values, y_predictions, c="k", alpha=0.4)
                
        ## labeling.
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title(label = f"{labels[i]}", fontsize = 20)
        plt.legend(loc='upper left')

## MSE
def MSE(y_true, post_pred, y_pred_name, num_idx = 1): 
    """get mean squared error (MSE). 

    Args:
        y_true (np.array): true y vector (test data).
        post_pred ([type]): [description]
        y_pred_name ([type]): [description]
        num_idx (int, optional): Number of groups. Defaults to 1.

    Returns:
        [type]: [description]
    """    
    
    ''' 
    y_true: true y values
    y_pred: predicted y values 
    y_pred_name: name of outcome in model. 
    num_idx: number of groups
    '''
    
    # setting up needed variables to index. 
    _, total = post_pred[y_pred_name].shape
    num_ypred = int(total/num_idx)
    
    # initialize the list & rolling
    MSE = []
    num_ypred_rolling = num_ypred
    
    # loop over number of idx
    for i in range(num_idx):
        # get current y_pred & y_true
        y_pred_tmp = post_pred[y_pred_name][:, num_ypred_rolling-num_ypred:num_ypred_rolling].mean(axis = 0)
        y_true_tmp = y_true[num_ypred_rolling-num_ypred:num_ypred_rolling]
        
        # compute MSE 
        MSE_tmp = np.square(np.subtract(y_true_tmp, y_pred_tmp)).mean() 
        
        # store MSE in list
        MSE.append(MSE_tmp) 
        
        # add to rolling window
        num_ypred_rolling += num_ypred

    return MSE 

# get residuals.
def get_resid(y_true, post_pred, y_pred_name, num_idx = 1):
    '''
    pretty much the same as above. 
    '''
    # setting up needed variables to index. 
    _, total = post_pred[y_pred_name].shape
    num_ypred = int(total/num_idx)
    
    # initialize the list & rolling
    MSE = []
    num_ypred_rolling = num_ypred
    
    error = []
    for i in range(num_idx):
        
        # get current y_pred & y_true & error. 
        y_pred_tmp = post_pred[y_pred_name][:, num_ypred_rolling-num_ypred:num_ypred_rolling].mean(axis = 0)
        y_true_tmp = y_true[num_ypred_rolling-num_ypred:num_ypred_rolling]
        error_tmp = [(true - pred) for true, pred in zip(y_true_tmp, y_pred_tmp)]
        
        # store error in list. 
        error.append(error_tmp) 
        
        # add to rolling window
        num_ypred_rolling += num_ypred
        
    return(error)