import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,OneHotEncoder


#Logic for Algorithm Creation
class RegressionModelPreparation:

    #Function use to Preparing Data for Processing
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


obj_alog=RegressionModelPreparation()