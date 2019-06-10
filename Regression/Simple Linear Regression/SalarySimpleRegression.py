import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

salarydf=pd.read_csv('Salary_Data.csv')
x=salarydf["YearsExperience"].values
y=salarydf["Salary"].values


plt.scatter(x,y,color="red")

reg= LinearRegression()
m=len(x)
x=x.reshape(m,1)
y=y.reshape(m,1)
reg.fit(x,y)
y_pred=reg.predict(x)
plt.plot(x,y_pred,color="blue")
r2=reg.score(x,y_pred)
print(r2)
plt.show()