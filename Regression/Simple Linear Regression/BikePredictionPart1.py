import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

dataset=pd.read_csv('../Regression/bike_sharing.csv')

x=dataset["temp"].values
y=dataset["cnt"].values

plt.scatter(x,y,color="red")

reg=LinearRegression()
m=len(x)
x=x.reshape(m,1)
y=y.reshape(m,1)
#linear equation Y= a *X + b.
reg.fit(x,y)
y_pred=reg.predict(x)
plt.plot(x,y_pred,color='blue')
r2=reg.score(x,y_pred)
plt.title("Bike Sharing Prediction")
plt.xlabel("Temperature")
plt.ylabel("Count")

#Equation coefficient and Intercept
print('Coefficient: \n', reg.coef_)
print('Intercept: \n', reg.intercept_)
print("R Sqaure Score",r2)
plt.show()