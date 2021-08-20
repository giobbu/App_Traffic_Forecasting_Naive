
import os
import pandas as pd
import numpy as np
from util import logging
import streamlit as st



def data_reader(path_file):
    data = pd.read_csv(path_file)
    nRow, nCol = data.shape
    logging.info("-- read data")
    return data


def streets_selection(THRESHOLD, DATAFRAME):

    DATAFRAME_STREETS = DATAFRAME.iloc[:,1:] # without index
    LST_STREETS = list(DATAFRAME_STREETS.columns.values)

    LST_MEAN_VALUE = []
    LST_STREETS_NEW = []

    for street in LST_STREETS:
        values_street = DATAFRAME_STREETS[street]
        mean_value_street = np.mean(values_street)
        LST_MEAN_VALUE.append(mean_value_street)
        LST_STREETS_NEW.append(street)

    # create dataframe with street index column and avg value of flow/vel
    DF_MEAN = pd.DataFrame({'street_index': LST_STREETS_NEW, 'mean_value': LST_MEAN_VALUE})
    # select streets with avg value higher than threshold
    SLCT_STREETS = DF_MEAN[(DF_MEAN ['mean_value']>= THRESHOLD)] 
    SLCT_STREETS = SLCT_STREETS.sort_values(by=['street_index'])
    LST_SLCT_STREETS = list(SLCT_STREETS.street_index)

    logging.info("-- select streets with thresh: "+str(THRESHOLD))
    return LST_SLCT_STREETS



def feat_engin(MEAN_VALUE, dataframe):

    # select streets with avg flow/velocity higher than threshold "MEAN_VALUE"
    LST_SLCT_STREETS = streets_selection(MEAN_VALUE, dataframe)

    # convert timestamp into datetime column
    dataframe['Datetime'] = pd.to_datetime(dataframe['datetime'])
    # copy dataframe
    DATAFRAME_ = dataframe
    DATAFRAME_ = DATAFRAME_.drop(['datetime'],axis=1) 
    DATAFRAME_ = DATAFRAME_[DATAFRAME_.columns.intersection(LST_SLCT_STREETS)]


    timestamp = dataframe['datetime']
    logging.info("-- add time-based features")
    return DATAFRAME_, LST_SLCT_STREETS, timestamp



def sequence_in_out_naive(dataframe, INPUT, OUTPUT):
    X, y = list(), list()
    for i in range(len(dataframe)):
        # find the end of this pattern
        end_ix = i + INPUT
        out_end_ix = end_ix + OUTPUT
        # check if we are beyond the dataset
        if out_end_ix > len(dataframe):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = dataframe[i:end_ix, :], dataframe[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def data_loader_naive(dataframe, INPUT, OUTPUT, VALID_LENGTH, TEST_LENGTH, TIMESTAMP):
    X, Y =  sequence_in_out_naive(dataframe, INPUT, OUTPUT )

    X1, Y1 = X[: -VALID_LENGTH -TEST_LENGTH], Y[: -VALID_LENGTH -TEST_LENGTH]
    X2, Y2 = X[ : -TEST_LENGTH +23], Y[ : -TEST_LENGTH +23]
    X3, Y3 = X[-TEST_LENGTH +23 : ], Y[-TEST_LENGTH +23 : ]

    timestamp_test = TIMESTAMP[ -TEST_LENGTH : ]

    return X1, Y1, X2, Y2, X3, Y3, timestamp_test





