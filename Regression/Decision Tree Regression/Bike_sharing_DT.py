import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

#Splitting Model in train and Testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

#Implement Decision Regression Algorithm
from sklearn.tree import DecisionTreeRegressor
reg=DecisionTreeRegressor(random_state=0)
reg.fit(X_train,y_train)

#Predicting New Result
y_pred=reg.predict(X_test)
print("Actual Value:\n",y_test)
print("Predicted Value\n",y_pred)

#Calculate the accuracy from r2_square
from sklearn.metrics import r2_score
accurate=r2_score(y_test,y_pred)
print("Accurate Result:\n",accurate)

X_grid = np.arange(min(X_train), max(X_train), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

# scatter plot for original data
plt.scatter(X_train,y_train, color='red')

# plot predicted data
plt.plot(X_grid, reg.predict(X_grid), color='blue')

# specify title
plt.title("Bike vs Temp")

# specify X axis label
plt.xlabel("Temperture")

# specify Y axis label
plt.ylabel("Bike Count")

# show the plot
plt.show()