#@ Author Akbar Belif

# Import Class Library
import numpy as np
import pandas as pd
import csv
#import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


#Data Processing
class ClassificationRegressionModelPreparation:

    def data_processing(self,csv_path):

        #Read Csv file to dataset
        dataset = pd.read_csv(csv_path)
        old_lenght = len(dataset)
        print("Length:\n", old_lenght)
        #dataset_header=pd.DataFrame(dataset.head(n=0))

        #if dataset.isnull().sum().sum() > 0:
        print("DataFrame has Null Value.")

        for i in dataset.columns:
            if dataset[i].isnull().sum().sum() > 0:

                if (dataset[i].dtype == np.number):
                    col_index=dataset.columns.get_loc(i)
                    self.handle_null(col_index,"mean")
                else:
                    col_index = dataset.columns.get_loc(i)
                    self.handle_null(col_index,"most_frequency")

                    #obj_index=dataset.columns.getloc(i)
                    #print("Index of", i, " is ", obj_index)
                    #self.handle_null(obj_index,strategy)

        print("Completed Successfully")
        #  print("Object type\n",dataset.select_dtypes(include=[np.object]))
        # dep_column = input("Enter Dependence Column")
        # categorical_column = input("Enter Categorical")
        #
        # for i in dataset_header:
        #     if  i in categorical_column:
        #
        #         dataset=self.cat_encode(self,dataset,categorical_column)
        #         new_lenght=len(dataset)
        #         print("New Lenght: ",new_lenght)
        #
        # if len(dep_column) > 0:
        #     if  dep_column in dataset_header:
        #        dataset=self.handle_null(dep_column)
        # else:
        #     print("Dependent variable is must")

        # x_num_index = []
        # x_cat_index = []
        # X = pd.DataFrame()
        # y_index=dataset.columns.get_loc(Dep_Column)
        # Y=dataset.iloc[:,y_index: y_index + 1]
        #
        #
        # #Numerical Column
        # if len(Indep_Column_Number) > 0:
        #     for i in Indep_Column_Categorical:
        #         x_num_index.append(dataset.columns.get_loc(Indep_Column_Number[i]))
        #     X=dataset.iloc[:,x_num_index:x_num_index+1]
        #
        # #Categorocal Column
        # if len(Indep_Column_Categorical) > 0:
        #     if len(Indep_Column_Number) > 0:
        #         X=pd.concat(X,dataset.iloc[:,x_num_index:x_index +1],axis=1)
        #     else:
        #         for i in Indep_Column_Categorical:
        #             x_cat_index.append(dataset.column.get_loc(Indep_Column_Categorical[i]))
        #             X=dataset.iloc[:,x_cat_index:x_cat_index+1]
        #
        #
        #
        # if len(y_index) > 0:
        #    dataset=self.handle_null(y_index)
        #
        # if len(Indep_Column_Number) > 0:
        #     dataset=self.handle_null(x_index)
        #
        # if len(X_Categorical_index) > 0:
        #     dataset=self.stand_scaler(X_Categorical_index)
        #
        # self.Storing_data(self,dataset)

    # Handle Missing Data

    def handle_null(self,col_index,strategy):

            with SimpleImputer(missing_values=np.nan,strategy=strategy) as nan:
                new_column=nan.fit_transform(col_index)

            print("Null Resolved\n:", new_column)
            return new_column

    # Standard Scaling
    def stand_scaler(self,column):
        from sklearn.preprocessing import StandardScaler

        # formula z=(x-u)/s
        scX = StandardScaler()
        new_column = scX.fit_transform(column)
        #print("Standard Scaling Colum:\n", new_column)
        return new_column

    # Label Encoder
    def label_encode(self):

        from sklearn.preprocessing import LabelEncoder
        lbX = LabelEncoder()
        new_column = lbX.fit_transform(self)
        print("Label column:\n", new_column)
        return new_column

        # Dummies Categorical Variable
    def cat_encode(self,dataset,Indep_column=[]):

        for i in Indep_column:
            new_column=pd.get_dummies(dataset[i])
            dataset=dataset.drop(i,axis=1)
            dataset=pd.concat([dataset,new_column],axis=1)

        return dataset

    def Storing_data(self, clean_dataset):

        # Split dataset into Train and Testing
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(clean_dataset, test_size=0.2)

        # Load train and testing data into csv format
        train.to_csv('../Random_forest_Classification/Data/training.csv', header=True, index=None)
        test.to_csv('../Random_forest_Classification/Data/testing.csv', header=True, index=None)
        print("Done Cleaning and Storing Records")



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

obj_Classlog= ClassificationRegressionModelPreparation()