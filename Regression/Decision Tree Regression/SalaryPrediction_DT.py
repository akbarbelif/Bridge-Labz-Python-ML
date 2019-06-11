import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sal_dataset=pd.read_csv("Position_Salaries.csv")
X=sal_dataset.iloc[:,1:2].values
y=sal_dataset.iloc[:,2].values


#Fitting Decission Tree Regression to the DataSet
from sklearn.tree import DecisionTreeRegressor
dectree_reg=DecisionTreeRegressor(random_state= 0)
dectree_reg.fit(X,y)

#Predicting the new Result
y_pred=dectree_reg.predict(X)

#Visualising the Decision Tree Regression result
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='r')
#Predicting the New Grid Result
y_gridpred=dectree_reg.predict(X_grid)
plt.plot(X_grid,y_gridpred,color='b')
plt.title("Truth or Bluff (Decision Tree Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()









