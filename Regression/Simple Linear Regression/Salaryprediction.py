#refered from https://towardsdatascience.com/machine-learning-simple-linear-regression-with-python-f04ecfdadc13
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Import Csv File inot dataset
df=pd.read_csv('Salary_Data.csv')
x=df.iloc[:,:-1]#Array to Copy Last Column(Salary)
y=df.iloc[:,1]#Array to copy First Column(Year Experience)
print("Salary:\n",x)
print("Year Experience:\n",y)

#Split the Dataset into Training and Testing
from sklearn.model_selection import train_test_split
##test_size=1/3: we will split our dataset into 2 parts (ie:training set,test set)
#and the ratio of test set compare to dataset is 1/3
##train_size:if we use the test_size already, the rest of data will automatically be assigned to train_size.
##random-state:this is the seed for the random number generator.
#We can put an instance of the RandomState class as well.
#If we leave it blank or 0, the RandomState instance used by np.random will be used instead.
X_Train,X_Test,Y_Train,Y_Test=train_test_split(x,y,test_size=1/3, random_state=0)


#We already have the train set and test set, now we have to build the Regression Model:
#Fitting Simple Linear Reg
# ression to the Training set
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(X_Train,Y_Train)
##regressor = LinearRegression(): our training model which will implement the Linear Regression.
##regressor.fit: in this line, we pass the X_train which contains value of Year Experience and
#Y_train which contains values of particular Salary to form up the model. This is the training process.


#Visualizing the training set result
viz_train=plt
viz_train.scatter(X_Train,Y_Train,color='Red')
viz_train.plot(X_Train,regression.predict(X_Train),color='Blue')
viz_train.title("Salary vs Year of Experience(Training Set)")
viz_train.xlabel('Year of Experience')
viz_train.ylabel('Salary')
viz_train.show()

viz_test=plt
viz_test.scatter(X_Test,Y_Test,color='Red')
viz_test.plot(X_Test,regression.predict(X_Test),color='Blue')
viz_test.title("Salary vs Year of Experience(Test Set)")
viz_test.xlabel('Year of Experience')
viz_test.ylabel('Salary')
viz_test.show()
