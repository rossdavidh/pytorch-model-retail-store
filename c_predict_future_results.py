import os
import sys
import pickle

import pandas as pd
import torch
from torch.autograd import Variable
from scipy.stats import linregress

from a_load_dataframe import transform_columns,transform_numeric


def audit_command_line_inputs(argv):
    # e.g. python c_predict_future_results.py post_cutoff_sales_dataframe.p sales_prediction_network.p output_filename.csv 
    output_msg          = None
    inputs              = {}
    input_labels        = ['future_dates','trained_network','output_filename']
    if len(argv) != (1 + len(input_labels)):
        print('lenargv',len(argv),'leninputlabels',len(input_labels))
        output_msg = 'arguments should be: '
        for label in input_labels:
            output_msg += label
            output_msg += ', '
        output_msg      = output_msg[:-2] #chop off last ', '
    elif (argv[1][-2:] != '.p'):
        output_msg      = 'first argument must be the name of the saved dataframe of future dates'
    elif (argv[2][-2:] != '.p'):
        output_msg      = 'second argument must be the name of the saved trained network'
    elif (argv[3][-4:] != '.csv'):
        output_msg      = 'third argument must be the name of the output csv'
    else:
        for index,key in enumerate(input_labels):
            inputs[key] = argv[index+1]
    return output_msg,inputs




if __name__ == '__main__':
    output_msg,inputs = audit_command_line_inputs(sys.argv)
    if output_msg:
        print(output_msg)
        sys.exit(127)
    df                         = pickle.load(open(inputs['future_dates'],'rb'))
    model                      = pickle.load(open(inputs['trained_network'],'rb'))
    x_df                       = df.copy()
    if 'total_sales' in x_df.columns.values.tolist():
        del x_df['total_sales']
    x                          = Variable(torch.from_numpy(x_df.values).float())
    #predictions                = model(x)
    df['predictions']          = model(x).data.numpy()[:,0]
    slope, intercept, r_value, p_value, std_err = linregress(df['predictions'], df['total_sales'])
    print('R squared',r_value*r_value, 'p_value',p_value, 'std_err',std_err)
    df.to_csv('prediction_results.csv',columns=['predictions','total_sales','day_tot'])
