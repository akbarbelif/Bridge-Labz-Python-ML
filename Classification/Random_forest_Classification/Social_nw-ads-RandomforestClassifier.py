#Import Class Library
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt

from Classification.ClassificationRegression_DataProcessing import *
from Classification.ClassificationRegression_Model import *
from Classification.Classifier_LoadingnUnloadingModel import *

class Rand_forest_classifier:

    def __int__(self,input_data_path,train_data_path):
        try:

            #Data Processing

            ClassificationRegressionModelPreparation.Storing_data(self.input_data_path)

            dataset=pd.read_csv(self.train_data_path)
            print(dataset.head())
            new_lenght = len(dataset)
            print("New Length:\n", new_lenght)
            X_index1=dataset.columns.get_loc("Age")
            X_index2=dataset.columns.get_loc("EstimatedSalary")
            print(X_index1)

            X=dataset.iloc[:,[X_index1,X_index2]].values
            y=dataset.iloc[:,4].values

            #Splitting model into Training and testing
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

            #Standard Scaling
            X_train=ClassificationRegressionModelPreparation.stand_scaler(X_train)

            #Traning the Model using Random Forest Classifier
            ClassificationRegression_Model.Random_forest_classifier(X_train,y_train,X_test,y_test)
            pass

        except Exception as e:
            print(e)
            pass



input_path="../Deep-Learing/Classification/Input_data/Social_Network_Ads.csv"
training_path="../Deep-Learing/Classification/Random_forest_Classification/Data/training.csv"
RandomForestClassifier(input_path,training_path)

