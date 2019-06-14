import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Regression.Regression_DataProcessing import *
from Regression.Regression_Model import *

class SVR:

    RegressionModelPreparation.Storing_data('../Input_data/Position_Salaries.csv')
    #import training Model
    dataset=pd.read_csv('../Support_Vector_Regression/Data/training.csv',sep=",",delimiter= None)
    X=dataset.iloc[:,1:2]
    y=dataset.iloc[:,-1:]

    #print("X-Axis\n",X)
    #print("Y-AXis\n",y)

    ##Splitting model into Training and testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    #Support Vector Regression Model Training
    y_pred=Regression_Model.SupportVector_reg(X,y)
    print("Y Predicted Value:\n",y_pred)

    #Accuracy Calculation using r2 score
    Accuracy=Regression_Model.r2_cal(y,y_pred)
    print("Accuary Between True value and Predicted Value:\n",Accuracy)


    # #Data preprocessing
    # #******Step 1**********#
    # def handle_null(column):
    #     #Handle Missing Data
    #     if dataset.isnull().sum().sum() > 0:
    #         from sklearn.impute import  SimpleImputer
    #         #set strategy to mean/median/most-Frequent
    #         print("Check the Salary for NUll Value")
    #         impute=SimpleImputer(missing_values=np.nan,strategy='mean')
    #         new_column=impute.fit_transform(column)
    #         print("Null Resolved\n:",new_column)
    #         return new_column
    #
    # #******Step 2**********#
    # #Standard Scaling
    # def stand_scaler(column):
    #     from sklearn.preprocessing import StandardScaler
    #     #formula z=(x-u)/s
    #     scX=StandardScaler()
    #     new_column=scX.fit_transform(column)
    #     print("Standard Scaling Colum:\n",new_column)
    #     return new_column
    #
    # #******Step 3**********#
    #     #Label Encoder
    # def label_encode(column):
    #     from sklearn.preprocessing import LabelEncoder
    #     lbX=LabelEncoder()
    #     new_column=lbX.fit_transform(column)
    #     print("Label column:\n",new_column)
    #     return new_column





