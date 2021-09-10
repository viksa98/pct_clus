import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import *
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':

    data_path = os.getcwd() + '/data/'
    filename = 'BO_truncated_mso_2018.pcl'

    df = load_hecat_data(path = data_path, filename = filename)

    # df = df[df['truncated']==0]
#     vektor = df['truncated'].to_numpy()
    df = drop_unnecessary_columns(df)
    df = rename_columns(df)
    df = transform_entry_date(df)
    df = df[df['duration']>0]
    
    columns_to_keep = ['id', 'truncated', 'Age', 'Months_of_work_experience', 'Entry_month_sin', 'Entry_day_sin',
       'Entry_month_cos', 'Entry_day_cos', 'Gender', 'Education_category', 'Dissabilities',
        'Reason_for_PES_entry', 'eApplication', 'Employment_plan_status', 'Employability_assessment',
        'Employment_plan_ready', 'duration']

    continuous = ['Age', 'Months_of_work_experience', 'Entry_month_sin', 'Entry_day_sin',
           'Entry_month_cos', 'Entry_day_cos']

    df = df[columns_to_keep]
    
#     one_hot_encoded_data = pd.get_dummies(new_df, columns = [column for column in new_df.columns if column not in continuous+['eApplication', 'Gender', 'duration']])
#     final_df = one_hot_encoded_data.copy()
    
    le = preprocessing.LabelEncoder()
    cols = ['eApplication', 'Gender']
    for col in cols:
        df[col] = le.fit_transform(df[col].values)
        
    for_scaling = ['Age', 'Months of work experience']

    scaler = MinMaxScaler() 
    scaled_values = scaler.fit_transform(df[[col for col in df.columns if col in for_scaling]]) 
    df.loc[:,[col for col in df.columns if col in for_scaling]] = scaled_values
    
    df_test = df.sample(frac=0.2)
    df_train = df.drop(df_test.index)
    
    df_train.to_csv('train_hecat.csv')
    df_test.to_csv('test_hecat.csv')
    df.to_csv('full_hecat.csv')