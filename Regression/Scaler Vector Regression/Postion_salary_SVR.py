import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Store Data into training and testing csv
dataset=pd.read_csv('Position_Salaries.csv')
old_lenght=len(dataset.columns)

#Split dataset into Train and Testing
from sklearn.model_selection import train_test_split
train,test=train_test_split(dataset,test_size=0.2)

#Export train and testing data into csv format
train.to_csv('training.csv',header=True,index=None)
test.to_csv('testing.csv',header=True,index=None)

#import training Model
dataset=pd.read_csv('training.csv',sep=",",delimiter= None)
X=dataset.iloc[:,1:2]
y=dataset.iloc[:,-1:]


#Data preprocessing
#******Step 1**********#
def handle_null(column):
    #Handle Missing Data
    if dataset.isnull().sum().sum() > 0:
        from sklearn.impute import  SimpleImputer
        #set strategy to mean/median/most-Frequent
        print("Check the Salary for NUll Value")
        impute=SimpleImputer(missing_values=np.nan,strategy='mean')
        new_column=impute.fit_transform(column)
        print("Null Resolved\n:",new_column)
        return new_column

#******Step 2**********#
    #Standard Scaling
def stand_scaler(column):
    from sklearn.preprocessing import StandardScaler
    #formula z=(x-u)/s
    scX=StandardScaler()
    new_column=scX.fit_transform(column)
    print("Standard Scaling Colum:\n",new_column)
    return new_column

#******Step 3**********#
    #Label Encoder
def label_encode(column):
    from sklearn.preprocessing import LabelEncoder
    lbX=LabelEncoder()
    new_column=lbX.fit_transform(column)
    print("Label column:\n",new_column)
    return new_column








