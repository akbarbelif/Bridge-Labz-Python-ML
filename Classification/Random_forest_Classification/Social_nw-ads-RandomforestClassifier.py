#Import Class Library
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt

from Classification.ClassificationRegression_DataProcessing import *
from Classification.ClassificationRegression_Model import *

class Rand_forest_classifier:

    ClassificationRegressionModelPreparation.Storing_data('../Input_data/Social_Network_Ads.csv')
    dataset=pd.read_csv('../Random_forest_Classification/Data/training.csv')
    print(dataset.head())
    new_lenght = len(dataset)
    print("New Length:\n", new_lenght)
    X_index1=dataset.columns.get_loc("Age")
    X_index2=dataset.columns.get_loc("EstimatedSalary")
    print(X_index1)

    X=dataset.iloc[:,[X_index1,X_index2]].values
    y=dataset.iloc[:,4].values
    # print("X Axis\n",X)
    #print("Y Axis \n",y)

    ##Splitting model into Training and testing
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    #print("Below Scaling",X_train)

    #Standard Scaling
    X_train=ClassificationRegressionModelPreparation.stand_scaler(X_train)
    #print("after scaling:\n",X_train)

    y_pred=ClassificationRegression_Model.Decision_tree_classifier(X_train,y_train,X_test,y_test)
    print("\nY_predicted Value:\n",y_pred)
    print("\nY_Ground True Value:\n", y_test)

    # Accuracy Calculation using confusion_matrix score
    Accuracy = ClassificationRegression_Model.confusion_matix(y_test, y_pred)
    print("\nAccuary Between True value and Predicted Value:\n", Accuracy)