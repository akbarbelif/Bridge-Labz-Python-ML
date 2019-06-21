# Pandas Class Library Installation
# sudo apt-get install python3-pandas
#Raise to power element of second element
import pandas as pd
import numpy as np

a=pd.Series([0,1,2,3,4,5,6]) 
rais=pd.Series([])
print("Original :\n",a)
#rais=a[2] ** a[3]
for i in range(0,len(a)):
    rais=a[i] ** a[3]
    print(rais)

#rais=[a[i]**a[j] for j in range(1,len(a)) for i in range(0,len(a))]
result=pd.Series(rais)
print("Old For loop way:",result)


df=pd.DataFrame({'X':[0,1,2,3,4,5,6]})
print("DataFrames Panda way:",df)


first_arr=np.arange(6)
sec_arr=np.arange(6)
pows=np.power(first_arr,sec_arr)
print("using numpy",pows)

