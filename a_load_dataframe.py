import sys
import pickle
import datetime
import calendar

import pandas as pd
import torch
from torch.autograd import Variable


def audit_command_line_inputs(argv):
    # e.g. python a_load_dataframe.py scaled_sales.csv sales_dataframe.p
    output_msg          = None
    inputs              = {}
    input_labels        = ['scaled_input_csv','dataframe_save_filename']
    if len(argv) != (1 + len(input_labels)):
        print('lenargv',len(argv),'leninputlabels',len(input_labels))
        output_msg = 'arguments should be: '
        for label in input_labels:
            output_msg += label
            output_msg += ', '
        output_msg      = output_msg[:-2] #chop off last ', '
    elif (argv[1][-4:] != '.csv'):
        output_msg      = 'first argument must be the name of the scaled sales csv file'
    elif (argv[2][-2:] != '.p'):
        output_msg      = 'second argument must be the name to save the dataframe as'
    else:
        for index,key in enumerate(input_labels):
            if key == 'max_epochs':
                inputs[key] = int(argv[index+1])
            else:
                inputs[key] = argv[index+1]
    return output_msg,inputs

def convert_categorical(df,col_name,cat_cols_to_drop):
    values = df[col_name].unique()
    print(col_name,df.groupby(col_name).salesdate.nunique())
    one_hot = pd.get_dummies(df[col_name])
    for v in values:
        df[col_name+':'+v] = one_hot[v]
    cat_cols_to_drop.append(col_name)
    return df,cat_cols_to_drop

def convert_numeric(df,col_name,min_value=None,max_value=None,range_value=None):
    if not min_value:
        min_value = min(df[col_name])
    if not max_value:
        max_value = max(df[col_name])
    if not range_value:
        range_value = max_value - min_value
    df[col_name] = df[col_name].apply(lambda x: (x - min_value) / range_value)
    return df,min_value,max_value

def transform_numeric(df,minmax_values=None):
    if not minmax_values:
        minmax_values = {'dotm':[None,None],'daysleftinmonth':[None,None],'daysleftinmonth':[None,None],'day_tot':[None,None]}
    for numeric_column in minmax_values.keys():
        mi                            = minmax_values[numeric_column][0]
        ma                            = minmax_values[numeric_column][1]
        df,min_value,max_value        = convert_numeric(df,numeric_column,min_value=mi,max_value=ma)
        minmax_values[numeric_column] = [min_value,max_value]
    return df,minmax_values

def transform_csv_into_df(scaled_input_csv):
    df                         = pd.read_csv(scaled_input_csv)
    print('total sales non-numeric values: ',df['total_sales'].isnull().sum(),'numeric values: ',df['total_sales'].count())
    df                         = transform_columns(df)
    df,minmax_values           = transform_numeric(df)
    for colname in minmax_values.keys():
        if minmax_values[colname][1] > 1:
            print(colname,' has a max greater than 1: ',minmax_values[colname][1])
    return df,minmax_values

def transform_columns(df):
    cat_cols_to_drop           = ['notes']
    df['notes']                = df['notes'].fillna('')
    df['salesdatetime']        = pd.DatetimeIndex(df['salesdate'])
    cat_cols_to_drop.append('salesdatetime')
    print('sales dates min: ',min(df['salesdatetime']),' and max: ',max(df['salesdatetime']))

    dayOfWeek={0:'1Monday', 1:'2Tuesday', 2:'3Wednesday', 3:'4Thursday', 4:'5Friday', 5:'6Saturday', 6:'7Sunday'}
    df['weekday']              = df['salesdatetime'].dt.dayofweek.map(dayOfWeek)
    df['weekday']              = df['weekday'].astype('category')
    df,cat_cols_to_drop        = convert_categorical(df,'weekday',cat_cols_to_drop)

    monthNames = {1:'01Jan',2:'02Feb',3:'03Mar',4:'04Apr',5:'05May',6:'06Jun',7:'07Jul',8:'08Aug',9:'09Sep',10:'10Oct',11:'11Nov',12:'12Dec'}
    df['month']                = df['salesdatetime'].dt.month.map(monthNames)
    df['month']                = df['month'].astype('category')
    df,cat_cols_to_drop        = convert_categorical(df,'month',cat_cols_to_drop)

    df['dotm']                                    = df['salesdatetime'].dt.day
    df['daysleftinmonth']                         = df['salesdatetime'].dt.daysinmonth - df['dotm']
    df['endofmonth']                              = 0
    df.loc[df['daysleftinmonth']==0,'endofmonth'] = 1
    print('nbr of end of month days: ',sum(df['endofmonth']))

    df['week']                 = '1first'
    df.loc[df.dotm>7, 'week']  = '2second'
    df.loc[df.dotm>14, 'week'] = '3third'
    df.loc[df.dotm>21, 'week'] = '4fourth'
    df.loc[df.dotm>28, 'week'] = '5fifth'
    df['week']                 = df['week'].astype('category')
    df,cat_cols_to_drop        = convert_categorical(df,'week',cat_cols_to_drop)

    df['day_tot']              = df['salesdatetime'] - datetime.datetime.strptime("2006-12-31","%Y-%m-%d")
    df['day_tot']              = df['day_tot'].dt.days
    print('days total min: ',min(df['day_tot']),' max: ',max(df['day_tot']))

    df['location']                                                                              = 'third'
    df.loc[df['salesdatetime']<datetime.datetime.strptime("2018-01-31","%Y-%m-%d"),'location']  = 'second'
    df.loc[df['salesdatetime']<datetime.datetime.strptime("2012-01-31","%Y-%m-%d"),'location']  = 'first'
    df['location']             = df['location'].astype('category')
    df,cat_cols_to_drop        = convert_categorical(df,'location',cat_cols_to_drop)

    df['holiday']                                                                                              = 'none'
    df['holiday'].loc[(df['month']=='01Jan') & (df['dotm']==1)]                                                = 'NewYearsDay'
    df['holiday'].loc[df['notes'].str.contains('aster')]                                                       = 'Easter'
    df['holiday'].loc[(df['month']=='05May') & (df['weekday']=='1Monday') & (df['dotm']>24)]                   = 'MemorialDay'
    df['holiday'].loc[(df['month']=='07Jul') & (df['dotm']==4)]                                                = 'July4th'
    df['holiday'].loc[(df['month']=='09Sep') & (df['weekday']=='1Monday') & (df['week']=='1first')]            = 'LaborDay'
    df['holiday'].loc[(df['month']=='10Oct') & (df['dotm']==31)]                                               = 'Halloween'
    df['holiday'].loc[(df['month']=='11Nov') & (df['dotm']==1)]                                                = 'DayOfTheDead'
    df['holiday'].loc[(df['month']=='11Nov') & (df['weekday']=='4Thursday') & (df['week']=='4fourth')]         = 'Thanksgiving'
    df['holiday'].loc[(df['month']=='11Nov') & (df['weekday']=='5Friday') & (df['dotm']>22) & (df['dotm']<30)] = 'BlackFriday'
    df['holiday'].loc[(df['month']=='12Dec') & (df['dotm']==24)]                                               = 'ChristmasEve'
    df['holiday'].loc[(df['month']=='12Dec') & (df['dotm']==25)]                                               = 'Christmas'
    df['holiday'].loc[(df['month']=='12Dec') & (df['dotm']==26)]                                               = 'BoxingDay'
    df['holiday'].loc[(df['month']=='12Dec') & (df['dotm']==31)]                                               = 'NewYearsEve'
    df['holiday'] = df['holiday'].astype('category')
    df,cat_cols_to_drop        = convert_categorical(df,'holiday',cat_cols_to_drop)

    df['tickets']                                         = 0
    df['tickets'].loc[df['notes'].str.contains('tix')]    = 1
    df['tickets'].loc[df['notes'].str.contains('ticket')] = 1

    df['ticket_accounting']                                                                              = 0
    df.loc[df['salesdatetime']>datetime.datetime.strptime("2018-02-28","%Y-%m-%d"),'ticket_accounting']  = 1
    print('nbr days with new ticket accounting system: ',sum(df['ticket_accounting']))

    df['anniversary']                                              = 0
    df['anniversary'].loc[df['notes'].str.contains('anniversary')] = 1
    print('anniversary dates found: ',df['anniversary'].sum())

    df['corset']                                         = 0
    df['corset'].loc[df['notes'].str.contains('corset')] = 1
    df['corset'].loc[df['notes'].str.contains('trunk')]  = 1
    print('corset events found: ',df['corset'].sum())

    df['goth_ball']                                              = 0
    df['goth_ball'].loc[df['notes'].str.contains('gothball')]    = 1
    df['goth_ball'].loc[df['notes'].str.contains('goth ball')]   = 1
    df['goth_ball'].loc[df['notes'].str.contains('gothic ball')] = 1
    print('goth ball events found: ',df['goth_ball'].sum())

    df['sale']                                             = 0
    df['sale'].loc[df['notes'].str.contains('sale event')] = 1
    print('sale events found: ',df['sale'].sum())

    df['tfw']                                                   = 0
    df['tfw'].loc[df['notes'].str.contains('tax free weekend')] = 1
    df['tfw'].loc[df['notes'].str.contains('tax-free weekend')] = 1
    df['tfw'].loc[df['notes'].str.contains('taxfree weekend')]  = 1
    print('tax free weekend days found: ',df['tfw'].sum())

    df['event']                                        = 0
    df['event'].loc[df['notes'].str.contains('event')] = 1
    df['event'].loc[df['corset']==1]                   = 0
    df['event'].loc[df['anniversary']==1]              = 0
    df['event'].loc[df['sale']==1]                     = 0
    df['event'].loc[df['goth_ball']==1]                = 0
    df['event'].loc[df['tfw']==1]                      = 0
    print('misc other events found: ',df['event'].sum())

    for col_name in cat_cols_to_drop:
        df = df.drop(col_name,axis=1)
    return df


if __name__ == '__main__':
    output_msg,inputs = audit_command_line_inputs(sys.argv)
    if output_msg:
        print(output_msg)
        sys.exit(127)
    df,minmax_values = transform_csv_into_df(inputs['scaled_input_csv'])
    pickle.dump(minmax_values,open('minmax_'+inputs['dataframe_save_filename'],'wb'))
    df_before        = df.copy().loc[df['salesdate'] < '2018-06-01']
    df_before        = df_before.drop('salesdate',axis=1)
    pickle.dump(df_before,open('pre_cutoff_'+inputs['dataframe_save_filename'],'wb'))
    df_after         = df.copy().loc[df['salesdate'] >= '2018-06-01']
    df_after         = df_after.drop('salesdate',axis=1)
    pickle.dump(df_after,open('post_cutoff_'+inputs['dataframe_save_filename'],'wb'))
