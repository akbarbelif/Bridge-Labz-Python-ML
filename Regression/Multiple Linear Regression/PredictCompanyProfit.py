import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

profit_df=pd.read_csv("50_Startups.csv")
#X has more Independent Feature as Compare to Y Dependent Feature
x=profit_df.iloc[:,:-1]
y=profit_df.iloc[:,4]

state=pd.get_dummies(x['State'],drop_first=True)
x=x.drop('State',axis=1)
x=pd.concat([x,state],axis=1)



reg=LinearRegression()
# m=len(x)
# x=x.reshape(m,1)
# y=y.reshape(m,1)
reg.fit(x,y)
y_pred=reg.predict(x)
plt.plot(x,y_pred)
r2=reg.score(x,y_pred)
print(r2)
plt.show()
