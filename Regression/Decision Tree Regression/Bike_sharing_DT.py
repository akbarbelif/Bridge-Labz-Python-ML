import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


bike_ds=pd.read_csv('bike_sharing.csv')
X=bike_ds.iloc[:,15:16].values
y=bike_ds.iloc[:,16].values
#print(X)
#print(y)


from sklearn.tree import DecisionTreeRegressor
decreg=DecisionTreeRegressor(random_state=0)
decreg.fit(X,y)

#Predicting New Result
y_pred=decreg.predict(X)
#print(y_pred)


#visualising the Regression Results
plt.scatter(X,y,color='r')
X_grid=np.arange(min(X),max(X),0.01)
#print(X_grid)
X_grid=X_grid.reshape((len(X_grid),1))
#Predicting New Grid
y_predgrid=decreg.predict(X_grid)
plt.plot(X_grid,y_predgrid,color='b')
plt.title("Predict Number Of Bikes")
plt.xlabel("Register")
plt.ylabel("Count")
#print(X_grid)
plt.show()
