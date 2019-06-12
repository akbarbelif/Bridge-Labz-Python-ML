import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Regression.Regression_DataProcessing import  *

#import csv File into Dataset Model
class ControllerRandomBike:

    def Importcsvfile(self):
        simple=RegressionModelPreparation()
        dataset=simple._datapreprocessing('bike_sharing.csv')
        print(dataset.head())
       #print("Testing")


obj=ControllerRandomBike()
obj.Importcsvfile()
