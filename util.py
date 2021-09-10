import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import torch

def load_hecat_data(path, filename):
#    filename = 'data.pcl'

    with open(path+'/'+filename, 'rb') as handle:
        b = pickle.load(handle)
    df=pd.DataFrame(b)
    return df

def drop_unnecessary_columns(hecat_df):
    hecat_df = hecat_df.drop(columns = ['DatumObdobja', 'MeseciBrezpos', 'DatumIzpisaBO', 'IDizpisaBO', 'Razvrstitev ZRSZ', 'PrejemnikDNDP', 'PrejemnikCSD', 'IdIndikatorPrometa', 'OEN', 'IDUpEnote', 'IzdelanZN', 'mso_from', 'mso_to', 'Unnamed: 0'])
    return hecat_df

def transform_entry_date(hecat_df):
    if 'Entry_date' in hecat_df.columns:
        hecat_df['Entry_month'] = hecat_df['Entry_date'].dt.month
        hecat_df['Entry_day'] = hecat_df['Entry_date'].dt.day
        hecat_df['Entry_month_sin'] = np.sin(2*np.pi*hecat_df['Entry_month']/max(hecat_df['Entry_month']))
        hecat_df['Entry_day_sin'] = np.sin(2*np.pi*hecat_df['Entry_day']/max(hecat_df['Entry_day']))
        hecat_df['Entry_month_cos'] = np.cos(2*np.pi*hecat_df['Entry_day']/max(hecat_df['Entry_day']))
        hecat_df['Entry_day_cos'] = np.cos(2*np.pi*hecat_df['Entry_day']/max(hecat_df['Entry_day']))
        hecat_df = hecat_df.drop(columns = ['Entry_date', 'Entry_month', 'Entry_day'])
        return hecat_df
    return hecat_df

def rename_columns(df):
    df = df.rename(columns={"idosebe": "id", "StarostLeta": "Age", "MeseciDelDobe": "Months_of_work_experience", "IDSpola": "Gender", "IDObcine": "Municipality", "IDDrzave": "Country", "IDpoklicaSKP08": "Profession (ESCO)", "IdInvalidnosti": "Dissabilities", "DatumVpisaBO": "Entry_date", "IDVpisaBO": "Reason_for_PES_entry", "ePrijava": "eApplication", "IDKlasiusProgram": "Profession_program", "IDKlasiusP": "Specific_profession_category", "IDklasiusSRV": "Education_category", "IDStanjaZN": "Employment_plan_status", "IDZaposljivosti": "Employability_assessment", "IDPrenehanjaDR": "Employment_plan_ready"})
    return df
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministick = True
    torch.backends.cudnn.benchmark = False


    
