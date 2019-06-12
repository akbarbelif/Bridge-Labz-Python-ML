import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#case 1
#X has more Independent Feature as Compare to Y Dependent Feature
dataset=pd.read_csv("50_Startups.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

print(dataset.head())

#Encoding categoical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder=LabelEncoder()
x[:,3]=label_encoder.fit_transform(x[:,3])
#String Category Column wise
onehotencoder = OneHotEncoder(categorical_features = [3])
x=onehotencoder.fit_transform(x).toarray()

#Avoiding Dummmy Variable Trap
x=x[:,1:]

#Spltting tge model in to Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Linear Regression Class
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)
#predict  the output variable
y_pred=reg.predict(X_test)

print("Y_train",y_test)
print("y Pred",y_pred)
# m=len(x)
# x=x.reshape(m,1)
# y=y.reshape(m,1)
'''color=np.random.randint(4)
reg.fit(x, y)
y_pred = reg.predict(x)

plt.plot(x,y_pred)
print(y_pred)
#plt.xlabel(i)
plt.ylabel("Profit")
plt.title(" Against Profit")
plt.show()
'''
#Equation coefficient and Intercept
print('Coefficient: \n', reg.coef_)
print('Intercept: \n', reg.intercept_)
#ordiratory leanear square to find the least sum (y-y`)2 =>min
r2=reg.score(X_test,y_pred)
print('R Sqaure Score', r2)


