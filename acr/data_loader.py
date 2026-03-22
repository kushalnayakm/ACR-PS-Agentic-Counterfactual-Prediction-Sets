import pandas as pd
import numpy as np
import urllib.request
import os
import zipfile

def get_adult_data():
    """Manual robust download of Adult Census dataset"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    filename = "adult_data_raw.csv"
    
    if not os.path.exists(filename):
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, filename)
    
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational_num', 'marital_status', 'occupation',
                    'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours_per_week', 'native_country',
                    'income']
    
    data = pd.read_csv(filename, names=column_names, sep=', ', engine='python')
    
    # Preprocess similar to DiCE
    data = data.replace('?', np.nan).dropna()
    data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)
    
    # Simple groupings to match DiCE reference
    data = data[['age', 'workclass', 'education', 'marital_status', 'occupation', 'race', 'gender', 'hours_per_week', 'income']]
    
    return data
