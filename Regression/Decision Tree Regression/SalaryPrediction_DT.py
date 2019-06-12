import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#spliting model into Train and testing Model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Fitting Decission Tree Regression to the DataSet
from sklearn.tree import DecisionTreeRegressor
reg=DecisionTreeRegressor(random_state= 0)
reg.fit(X_train,y_train)

#Predicting the new Result
#y_pred=reg.predict(np.array([6.5]).reshape(-1,1))
y_pred=reg.predict(X_test)

#check the accuracy by r2_score
from sklearn.metrics import r2_score
accuracy=r2_score(y_test,y_pred)
print("Accuracy:\n",accuracy)
print("Actual Value:\n",y_test)
print("Predict Value:\n",y_pred)

#Visualising the Decision Tree Regression result
X_grid=np.arange(min(X_train),max(X_train),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X_train,y_train,color='r')
#Predicting the New Grid Result
y_gridpred=reg.predict(X_grid)
y_pred=reg.predict(np.array([9.5]).reshape(-1,1))
print("Preduction of employe having 6.5 Expienece salary:",y_pred)
plt.plot(X_grid,y_gridpred,color='b')
plt.title("Truth or Bluff (Decision Tree Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()









