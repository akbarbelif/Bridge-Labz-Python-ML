#@ Author Akbar Belif

# Import Class Library
import numpy as np
import pandas as pd
import csv
#import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

#Data Processing
class ClassificationRegressionModelPreparation:

    def __init__(self,input_data_path,outputpath):

       self.csv_file_path=input_data_path
       self.outputpath=outputpath
       self.data_processing()


    def data_processing(self):

        #Read Csv file to dataset
        dataset = pd.read_csv(self.csv_file_path)
        df_data=pd.DataFrame(dataset)
        old_lenght = len(dataset.columns)
        print("Length:\n", old_lenght)

        #Handle Null Value
        for i in df_data.columns:
            if df_data[i].isnull().sum().sum() > 0:
                print("DataFrame has Null Value.")
                if (df_data[i].dtype == np.number):
                    df_data=self.handle_null(df_data,i,"mean")
                else:
                    df_data=self.handle_null(df_data,i,"most_frequent")
                print("Null Resolved\n")


        #Categorical Processing / One Hot encoder
        if self.valid_check("Do you wanna Categorical Process(Yes/No):\n"):

            categorical_column = input("Enter Categorical Column Name:\n")
            if len(categorical_column) > 0:
                   df_data=self.cat_encode(df_data,categorical_column)
            else:
                print("Dataset dont have Categorical Column\n")

        #Standard Scaling
        if self.valid_check("Do you wanna Standard Scale(Yes/No):\n"):
            stand_scale_column = input("Stand Scale Column Name:\n")
            df_data=self.stand_scaler(df_data,stand_scale_column)

        print(df_data)
        new_lenght = len(df_data.columns)
        print(new_lenght)

        training_path,testing_path=self.outputpath
        self.Storing_data(df_data,training_path,testing_path)

    # Handle Missing Data
    def handle_null(self,dataset,index,strategy):

        col_index = dataset.columns.get_loc(index)
        imputer = SimpleImputer(missing_values=np.nan,strategy=strategy)
        imputer = imputer.fit(dataset.iloc[:, col_index:col_index + 1])
        dataset.iloc[:, col_index:col_index + 1] = imputer.fit_transform(dataset.iloc[:, col_index:col_index + 1])
        # with SimpleImputer(missing_values=np.nan,strategy=strategy) as nan:
        #     new_column=nan.fit_transform(self.col_index)
        return dataset

    # Standard Scaling
    def stand_scaler(self,dataset,stand_column=[]):
        scX = StandardScaler()
        col_index = []

        for i in stand_column.split(","):
            col_index.append(dataset.columns.get_loc(i))
        new_dataset = pd.DataFrame(scX.fit_transform(dataset.iloc[:, col_index]))
        for i in stand_column.split(","):
            dataset = dataset.drop(i, axis=1)
        dataset = pd.concat([dataset, new_dataset], axis=1)
        print("Standard/Feature Scaler Column Done Properly")
        return dataset

    # Label Encoder
    def label_encode(self):

        from sklearn.preprocessing import LabelEncoder
        lbX = LabelEncoder()
        new_column = lbX.fit_transform(self)
        print("Label column:\n", new_column)
        return new_column

    # Dummies Categorical Variable
    def cat_encode(self,dataset,cat_column):

        for i in cat_column.split(","):
            new_column = pd.get_dummies(dataset[i])
            dataset = dataset.drop(i, axis=1)
            dataset = pd.concat([dataset, new_column], axis=1)
            print("Categorial Column Done Properly")
            return dataset

    def Storing_data(self,clean_dataset,train_path,test_path):

        # Split dataset into Train and Testing
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(clean_dataset, test_size=0.2)
        # Load train and testing data into csv format
        train.to_csv(train_path, header=True, index=None)
        test.to_csv(test_path, header=True, index=None)
        print("Done Cleaning and Storing Records")


    def valid_check(self,input_str):
        yesno=input(input_str)
        if yesno  in ["Yes","y"]:
            return True
        else:
            return False


#obj_Classreg= ClassificationRegressionModelPreparation()