import sys
import pickle
import datetime
import calendar

import pandas as pd
import torch
from torch.autograd import Variable
from scipy.stats import linregress

def audit_command_line_inputs(argv):
    # e.g. python b_train_network.py pre_cutoff_sales_dataframe.p sales_prediction_network.p 10000
    output_msg          = None
    inputs              = {}
    input_labels        = ['dataframe_filename','model_save_filename','max_epochs']
    if len(argv) != (1 + len(input_labels)):
        print('lenargv',len(argv),'leninputlabels',len(input_labels))
        output_msg = 'arguments should be: '
        for label in input_labels:
            output_msg += label
            output_msg += ', '
        output_msg      = output_msg[:-2] #chop off last ', '
    elif (argv[1][-2:] != '.p'):
        output_msg      = 'first argument must be the name the dataframe was saved as'
    elif (argv[2][-2:] != '.p'):
        output_msg      = 'second argument must be the name to save the trained model as'
    elif (not argv[3].isdigit()):
        output_msg      = 'third argument must be an integer, the maximum number of training cycles to allow'
    else:
        for index,key in enumerate(input_labels):
            if key == 'max_epochs':
                inputs[key] = int(argv[index+1])
            else:
                inputs[key] = argv[index+1]
    return output_msg,inputs


def create_x_and_y(df,inputs):
    train             = df.sample(frac=0.8,random_state=200)
    test              = df.drop(train.index)
    x_train_df        = train.copy()
    x_test_df         = test.copy()
    y_train_df        = train.copy()
    y_test_df         = test.copy()
    for col_name in df.columns.values.tolist():
        #TODO: allow input parameter of what to use as the target (e.g. transform of total sales)
        if col_name == 'total_sales':
            del x_train_df[col_name]
            del x_test_df[col_name]
        else:
            del y_train_df[col_name]
            del y_test_df[col_name]
    # make all our values floats, as opposed to doubles or whatever else
    x_train           = Variable(torch.from_numpy(x_train_df.values).float())
    x_test            = Variable(torch.from_numpy(x_test_df.values).float())
    y_train           = Variable(torch.from_numpy(y_train_df.values).float(), requires_grad=False)
    y_test            = Variable(torch.from_numpy(y_test_df.values).float(), requires_grad=False)
    return x_train,x_test,y_train,y_test


if __name__ == '__main__':
    output_msg,inputs = audit_command_line_inputs(sys.argv)
    if output_msg:
        print(output_msg)
        sys.exit(127)
    df                            = pickle.load(open(inputs['dataframe_filename'],'rb'))
    x_train,x_test,y_train,y_test = create_x_and_y(df,inputs)
    N, D_in, H, D_out = len(x_train), len(x_train[0]), 25, 1
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )
    loss_fn                       = torch.nn.MSELoss(size_average=False)
    best_so_far                   = None
    learning_rate                 = 1e-4
    for t in range(inputs['max_epochs']):
        y_train_pred              = model(x_train)
        loss                      = loss_fn(y_train_pred, y_train)
        if t != 0 and (t & (t - 1) == 0): #i.e. t is a power of 2, check if we should stop
            y_test_pred           = model(x_test)
            loss_test             = loss_fn(y_test_pred,y_test)
            print('epoch',t,'loss on train data',loss.data.item(),' and on test data ',loss_test.data.item())
            y_test_pred_values    = y_test_pred.data.numpy()[:,0]
            y_test_values         = y_test.data.numpy()[:,0]
            if not best_so_far or loss_test.data.item() < best_so_far:
                best_so_far = loss_test.data.item()
            elif t < 100: #i.e. we have just gotten started here
                pass
            else: #i.e. it has not improved for the last 50% of t, meaning we should stop
                break
        model.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.data -= learning_rate * param.grad.data
    slope, intercept, r_value, p_value, std_err = linregress(y_test_pred_values, y_test_values)
    print('r_value',r_value, 'p_value',p_value, 'std_err',std_err)
    pickle.dump(model,open(inputs['model_save_filename'],'wb'))
