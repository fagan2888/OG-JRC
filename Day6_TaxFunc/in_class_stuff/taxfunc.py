# Import files
import pandas as pd
import pickle

# Read in the data
# TaxFunc_pkl = 'taxdata42.pkl'
# with open(TaxFunc_pkl, 'rb') as f:
#     dict_params = pickle.load(f, encoding='latin1')
micro_data = pickle.load(open('taxdata42.pkl', 'rb'))
