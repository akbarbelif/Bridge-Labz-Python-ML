# Pandas Class Library Installation
# sudo apt-get install python3-pandas
# Addition,Multiplication,Division,Substraction of two series number
import pandas as pd

a=pd.Series([2, 4, 6, 8, 10])
print(a)
b=pd.Series([1, 3, 5, 7, 9])
print(b)
c=a+b
print("Addition of two Series:\n",c)
c=a*b
print("Multiplication of two Series:\n",c)
c=a-b
print("Substraction of two Series:\n",c)
c=a/b
print("Division of two Series:\n",c)