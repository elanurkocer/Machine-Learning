import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('missing_data.csv')
#age = data[['age']]
#print(age)

from sklearn.impute import SimpleImputer

imputer  = SimpleImputer(missing_values=np.nan,strategy='mean')

age = data.iloc[:,1:4].values
print(age)

imputer = imputer.fit(age[:,1:4]) #The fit function is used to train
age[:,1:4] = imputer.transform(age[:,1:4])
print(age)

