# Pandas Class Library Installation
# sudo apt-get install python3-pandas
# Create Series using Pandas and convert into List and define its Type
import pandas as pd

a=pd.Series([1,2,3,4,5,7,8,9])
print(a)
print(type(a))
print(a.tolist())
print(type(a.tolist()))

