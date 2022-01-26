import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import *
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    
    columns_to_keep = ['id', 'truncated', 'Age', 'Months_of_work_experience', 'Entry_month_sin', 'Entry_day_sin',
       'Entry_month_cos', 'Entry_day_cos', 'Gender', 'Education_category', 'Profession_program', 'Dissabilities',
        'Reason_for_PES_entry', 'eApplication', 'Employment_plan_status', 'Employability_assessment', 'duration']
    continuous = ['Age', 'Months_of_work_experience', 'Entry_month_sin', 'Entry_day_sin',
            'Entry_month_cos', 'Entry_day_cos']
    cols = ['eApplication', 'Gender', 'Profession_program']
    for_scaling = ['Age', 'Months_of_work_experience']

    data_path = os.getcwd() + '/data'
    filename = 'BO_truncated_mso_2018.pcl'

    df = load_hecat_data(path = data_path, filename = filename)
    df = drop_unnecessary_columns(df)
    df = rename_columns(df)
    df = transform_entry_date(df)
    df = df[df['duration']>0]
    
    df = df[columns_to_keep]
    df['Profession_program'] = df['Profession_program'].astype('str').str.zfill(4).str.slice(stop=2)
    df['Profession_program'] = df['Profession_program'].loc[df['Profession_program']!='0000']
    
    le = preprocessing.LabelEncoder()
    for col in cols:
        df[col] = le.fit_transform(df[col].values)

    # df['Profession_program'] = df['Profession_program'].astype('str')


    scaler = MinMaxScaler() 
    scaled_values = scaler.fit_transform(df[[col for col in df.columns if col in for_scaling]]) 
    df.loc[:,[col for col in df.columns if col in for_scaling]] = scaled_values
    
    df_test = df.sample(frac=0.2, random_state = 11)
    df_train = df.drop(df_test.index)
    
    df_train = df_train.set_index('id')
    df_test = df_test.set_index('id')
    
    df_train.to_csv('./train_hecat.csv')
    df_test.to_csv('./test_hecat.csv')
    df.to_csv('./full_hecat.csv')