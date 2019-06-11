import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

#case 1
profit_ds=pd.read_csv("50_Startups.csv")
#X has more Independent Feature as Compare to Y Dependent Feature
x=profit_ds.iloc[:,:-1].values
y=profit_ds.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder=LabelEncoder()
x[:,3]=label_encoder.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x=onehotencoder.fit_transform(x).toarray()
#Avoid Dummy variable trap
x= x[:,1:]


#case 2
# dataset = pd.read_csv('50_Startups.csv')
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 4].values
#
# # Encoding categorical data
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder = LabelEncoder()
# X[:, 3] = labelencoder.fit_transform(X[:, 3])
# onehotencoder = OneHotEncoder(categorical_features = [3])
# X = onehotencoder.fit_transform(X).toarray()
# print(X)


# state=pd.get_dummies(x['State'],drop_first=True)
# x=x.drop('State',axis=1)
# x=pd.concat([x,state],axis=1)

reg=LinearRegression()
# m=len(x)
# x=x.reshape(m,1)
# y=y.reshape(m,1)
color=np.random.randint(4)
reg.fit(x, y)
y_pred = reg.predict(x)

plt.plot(x,y_pred)
print(y_pred)
#plt.xlabel(i)
plt.ylabel("Profit")
plt.title(" Against Profit")
plt.show()

#Equation coefficient and Intercept
print('Coefficient: \n', reg.coef_)
print('Intercept: \n', reg.intercept_)
#ordiratory leanear square to find the least sum (y-y`)2 =>min
r2=reg.score(x,y_pred)
print('R Sqaure Score', r2)


