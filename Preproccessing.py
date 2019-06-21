import matplotlib
import numpy as np
import pandas as pd
from pandas import read_csv
dataset = read_csv('mushroom.csv', header=None)


pd.set_option('display.max_columns', 30)
# print summary statistics on each attribute
print(dataset.describe())
# print the first 10 rows of data
print(dataset.head(5))

# count of the number of missing values on each of these columns
# checks for missing values
print((dataset[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]] == 0).sum())

# mark zero values as missing or NaN
print('Checking for missing values')
dataset[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]] = dataset[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]].replace(0, np.NaN)
# count the number of NaN values in each column
print(dataset.isnull().sum())
# print the first 10 rows of data
print(dataset.head(5))
#  remove all rows with missing data
dataset.dropna(inplace=True)
# summarize the number of rows and columns in the dataset
print(dataset.shape)

totalmissing=(dataset.isnull().sum()).sum()
print('Here')
print(totalmissing)


# print summary statistics on each attribute
print(dataset.describe(include='all'))
# print the first 10 rows of data
#print(dataset.head(5))
#dataset['veil-type'].value.counts()



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
#%matplotlib

print('Before data renaming')
print(dataset.describe())
dataset.columns = ["cap-shape","cap-surface","cap-color","bruises","odor","gill-attachment","gill-spacing", "gill-size","gill-color","stalk-shape"    ,"stalk-surface-above-ring","stalk-surface-below-ring    ","stalk-color-above-ring"    ,"stalk-color-below-ring", "veil-type",   "veil-color",    "ring-number","ring-type","spore-print-color","population","habitat","Class"]
print('After data renaming')
print(dataset.describe())
dataset = dataset.drop(['veil-type'], axis=1)
print('After dropping veil type')
print(dataset.describe())
dataset = dataset.apply(LabelEncoder().fit_transform)
X = dataset.iloc[:,1:]
Y = dataset['Class']

dataset.info()