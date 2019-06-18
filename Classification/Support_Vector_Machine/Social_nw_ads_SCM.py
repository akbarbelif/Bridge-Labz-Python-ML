#Import Class Library
import numpy    as np
import pandas   as pd
import matplotlib.pyplot as plt

from Classification.ClassificationRegression_DataProcessing import *
from Classification.ClassificationRegression_Model import *
from Classification.Classifier_LoadingnUnloadingModel import *

class SCM:

    try:
        def __init__(self,input_path,training_path,testing_path,model_path):
            self.input_data_path=input_path
            self.training_data_path=training_path
            self.testing_data_path=testing_path
            self.model_data_path=model_path
            outputpath=training_path,testing_path
            ClassificationRegressionModelPreparation.data_processing(self.input_data_path,outputpath)
#            csv_file_path=self.input_data_path
 #           obj_Modelpreparation=ClassificationRegressionModelPreparation(csv_file_path)
            #obj_Modelpreparation(self,self.input_data_path)
           # ClassificationRegressionModelPreparation(self,self.input_data_path)
            dataset=pd.read_csv(self.training_data_path)
            print(dataset.head())
            new_lenght = len(dataset)
            print("New Length:\n", new_lenght)
            X_index1=dataset.columns.get_loc("0")
            X_index2=dataset.columns.get_loc("1")
            y_index=dataset.columns.get_loc("Purchased")
            print(X_index1)
            X=dataset.iloc[:,[X_index1,X_index2]].values
            y=dataset.iloc[:,y_index:y_index+1].values

            ##Splitting model into Training and testing
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

            pred_reg=ClassificationRegression_Model.SCM_classifier(X_train,y_train,X_test,y_test)
            y_pred,reg_model=pred_reg
            print("\nY_predicted Value:\n",y_pred)
            print("\nY_Ground True Value:\n", y_test)

            # Accuracy Calculation using confusion_matrix score
            Accuracy = ClassificationRegression_Model.confusion_matix(y_test, y_pred)
            print("\nAccuary Between True value and Predicted Value:\n", Accuracy)

            ##Model Serialization (Unload)
            Classifier_Serialize_Deserialize_model.serializing_Model(reg_model,self.model_data_path)
            pass

    except Exception  as e:
        print (e)
        pass

model_path="../Support_Vector_Machine/Model"
input_path="../Input_data/Social_Network_Ads.csv"
training_path="../Support_Vector_Machine/Data/training.csv"
testing_path="../Support_Vector_Machine/Data/testing.csv"
obj_scm=SCM(input_path,training_path,testing_path,model_path)