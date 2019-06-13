#Data Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplt

#importing dataset
dataset=pd.read_csv('bike_sharing.csv')
dataset.drop(["instant"],axis=1)
X_index1=dataset.columns.get_loc("temp")
X_index2=dataset.columns.get_loc("atemp")
X_index3=dataset.columns.get_loc("hum")
X_index4=dataset.columns.get_loc("windspeed")
y_index=dataset.columns.get_loc("cnt")
#x=dataset.iloc[:,10:14] and #y=dataset.iloc[:,16]
X=dataset.iloc[:,[X_index1,X_index2,X_index3,X_index4]].values
y=dataset.iloc[:,y_index:(y_index+1)].values
print(dataset.head())
print("X axis \n",X)
print("Y axis \n",y)
#Check for null/NaN
# if y['cnt'].isnull().sum() > 0:
#     print("Taking care of null values of cnt column")
# y = y.fillna(y.mean())
#
#
# if X['temp'].isnull().sum() > 0:
#     print("Taking care of null values of temp column")
#     X = X.fillna(X.mean())

#Feature Scaler
from sklearn.preprocessing import StandardScaler
scX=StandardScaler()
x=scX.fit_transform(X)

#Spliting Model in training and Testing set
from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#import Linear Model Regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
#fitting training Data Model
reg.fit(X_train,y_train)
#Using test data Model to predict y cordinate
y_pred=reg.predict(X_test)
print("Test Value \n",y_test)
print("Predit value \n",y_pred)

#check the accuracy using r2_score metric
from sklearn.metrics import r2_score
accuracy=r2_score(y_test,y_pred)
print("Accuracy: \n",accuracy)

# x_season=bike_df["season"].values
# y_cnt=bike_df["cnt"].values

#print(x)
#print(y)
# dtdate=pd.get_dummies(x['dteday'],drop_first=True)
# x=x.drop('dteday',axis=1)
# x=pd.concat([x,dtdate],axis=1)
#print(x)

