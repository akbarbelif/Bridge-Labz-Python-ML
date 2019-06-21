#@ Author Akbar Belif

# Import Class Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


#Logic for Algorithm Creation
class RegressionModelPreparation:
    @staticmethod
    def Storing_data(csv_path):
        # Store Data into training and testing csv
        print(csv_path)
        dataset = pd.read_csv(csv_path,delimiter=None,sep=",")
        #old_lenght = len(dataset.columns)
        print(dataset.head())
        # Split dataset into Train and Testing
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(dataset, test_size=0.2)

        # Export train and testing data into csv format
        train.to_csv('/home/admin1/Desktop/Deep-Learing/Classification/Data/training.csv', header=True, index=None)
        test.to_csv('/home/admin1/Desktop/Deep-Learing/Classification/Data/testing.csv', header=True, index=None)


    # Data preprocessing
    # ******Step 1**********#
    # Handle Missing Data
    def handle_null(self,dataset,column):

        if dataset.isnull().sum().sum() > 0:
            from sklearn.impute import SimpleImputer
            # set strategy to mean/median/most-Frequent
            print("Check the Salary for NUll Value")
            impute = SimpleImputer(missing_values=np.nan, strategy='mean')
            new_column = impute.fit_transform(column)
            print("Null Resolved\n:", new_column)
            return new_column

    # ******Step 2**********#
    # Standard Scaling
    def stand_scaler(column):
        from sklearn.preprocessing import StandardScaler

        # formula z=(x-u)/s
        scX = StandardScaler()
        new_column = scX.fit_transform(column)
        print("Standard Scaling Colum:\n", new_column)
        return new_column

    # ******Step 3**********#
    # Label Encoder
    def label_encode(column):

        from sklearn.preprocessing import LabelEncoder
        lbX = LabelEncoder()
        new_column = lbX.fit_transform(column)
        print("Label column:\n", new_column)
        return new_column

'''''''''#Function use to Preparing Data for Processing
  def _datapreprocessing(self,csv_file_path):
        #assigning Missing variable
        missing_value=["na","n/a","","-"]
        dataset=pd.read_csv(csv_file_path,na_values=missing_value)

        #Check For missing Data and null
        # if(dataset.isnull().sum()>0):
        #     print("Model has Missing Data")
        # else:
        new_dataset=dataset

        # #Check If any Categorical Data Present
        cat_col=dataset.select_dtypes(include=['object']).columns
        if cat_col.dtype.name=='object':
            cat_index=[dataset.columns.get_loc(c) for c in cat_col if c in dataset]
            encoder_ds=new_dataset.iloc[:,cat_index]
            onehotcoder=OneHotEncoder
            for cat_loop in cat_index:
                onehotcoder.fit(encoder_ds)
                onehotlabel=onehotcoder.transform(encoder_ds).toarray()
                convertdata=pd.DataFrame(onehotlabel)
                ds=new_dataset.drop(cat_col,axis=1)
                dataset=pd.concat([ds,convertdata],axis=1)

        return dataset
'''''''''

obj_alog=RegressionModelPreparation()