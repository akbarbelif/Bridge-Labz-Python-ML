import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''from Regression.Regression_DataProcessing import  *

#import csv File into Dataset Model
class ControllerRandomBike:

    def Importcsvfile(self):
        simple=RegressionModelPreparation()
        dataset=simple._datapreprocessing('bike_sharing.csv')
        print(dataset.head())
       #print("Testing")


obj=ControllerRandomBike()
obj.Importcsvfile()'''

dataset_name=lambda dataname : print("Data_Name: \n",dataname.head())
dataset_training=lambda traindata:print("Training Data: \n",traindata.head())
dataset_testing=lambda testdata:print("Testing Data: \n",testdata.head())

#########import Data
dataset=pd.read_csv("Position_Salaries.csv")
old_length=len(dataset.columns)
#dataset_name(dataset)

#########Handle Categorical Data
# position=pd.get_dummies(dataset['Position'])
# dataset=dataset.drop('Position',axis=1)
# dataset=pd.concat([dataset,position],axis=1)
#dataset_name(dataset)


###########Splitting dataset into 2 differnt csv file training and testing
df_training=dataset.sample(frac=0.7)
df_testing=pd.concat([dataset,df_training]).drop_duplicates(keep=False)
#dataset_training(df_training)
#dataset_testing(df_testing)

##########exporting 70:30 to Training and Testing Data to csv file
df_training.to_csv('training.csv',header=True,index=None)
df_testing.to_csv('testing.csv',header=True,index=None)

##########import training data
dataset=pd.read_csv('training.csv')
new_length=len(dataset.columns)

dataset_name(dataset)
Y_index=dataset.columns.get_loc("Salary")
X_index=dataset.columns.get_loc("Level")
X=dataset.iloc[:,X_index:X_index+1]
y=dataset.iloc[:,Y_index:(Y_index+1)]

print("X-axis: ",X)
print("Y-axis :",y)

##########Check for Salary Yaxis NUll or nan value and assign Mean to it
if y['Salary'].isnull().sum() >0:
    print("Take Care of Salary having Null Value")
    y=y.fillna(y.mean())

###########Feature Scaling to equal Level for X cordinate
from sklearn.preprocessing import StandardScaler
scale_X=StandardScaler()
#X=scale_X.fit_transform(X)


###########Splitting model into Training and testing
from sklearn.model_selection import train_test_split
X_train,y_train,X_test,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

#print(X_train)
#print(y_test)

#Fitting Training Model in Regression Algorithm
from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor(n_estimators=300,random_state=0)
reg.fit(X_train,y_train.values.ravel())

#using Model to Predict
y_pred=reg.predict(X_test)

print("Original\n",y_test)
print("Predict \n",y_pred)
#Calcalute the Accuray by R2
from sklearn.metrics import r2_score
accuracy=r2_score(y_test,y_pred)
print("Accuracy Result:\n",accuracy)

'''''
#Dumping Model into pickle
import pickle
if(accuracy > 0.8):
    filename='RandomForestRegression.pkl'
    pkl_file=open(filename,'wb')
    model =pickle.dump(reg,pkl_file)

    #loading pickle model to  predict data from test_data.csv file
    pkl_file=open(filename,'rb')
    Reg_model_pkl=pickle.load(pkl_file)

    #import testdata csv
    dataset_testdata=pd.read_csv('test_data.csv')
    #Set X and Y cordinate
    X_testdata=dataset_testdata.iloc[:,(old_length-1):new_length]
    y_testdata=dataset_testdata.iloc[:,Y_index:(Y_index+1)]
    #predict Y using Pickle Model
    y_pred_testdata=Reg_model_pkl.predict(X_testdata)
    #calculate Accurary using r2
    accuracy=r2_score(y_testdata,y_pred_testdata)

    print("Accuracy Result by pickle Model :\n",accuracy)'''''