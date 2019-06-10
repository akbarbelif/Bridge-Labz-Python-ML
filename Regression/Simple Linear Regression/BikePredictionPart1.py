import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

bikedf=pd.read_csv('bike_sharing.csv')
x=bikedf["temp"].values
y=bikedf["cnt"].values

plt.scatter(x,y,color="red")

reg=LinearRegression()
m=len(x)
x=x.reshape(m,1)
y=y.reshape(m,1)
reg.fit(x,y)
y_pred=reg.predict(x)
plt.plot(x,y_pred,color='blue')
r2=reg.score(x,y_pred)
plt.title("Bike Sharing Prediction")
plt.xlabel("Temperature")
plt.ylabel("Count")
print(r2)
plt.show()