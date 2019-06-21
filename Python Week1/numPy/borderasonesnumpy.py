import numpy as np

a=np.ones((5,5))  
print("Original using Basic Tech :\n",a)
for i in range(1,4):
    for j in range(1,4):
        a[i][j]=0
print("After Update Border :\n",a)

a=np.ones((5,5))   
print("Original Build in Tech :\n",a)
#a[positive row(top):negative row(bottom),
#   postive column(left):negative column(right)]
a[1:-1,1:-1]=0
print("After Update Border :\n",a)

