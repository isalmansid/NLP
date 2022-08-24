import numpy as np
import pandas as pd
import joblib

df= pd.read_csv('mum.csv')

#import pickle
#pickle.dump(rf,open( 'placement_model.pkl','wb'))

import joblib

joblib.dump(df,'mum.pkl')

model = joblib.load('mum.pkl')

print('done')

