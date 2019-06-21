import numpy as np
def inversematrix(x):
    print("Original Matrix :",x)
    y=np.linalg.inv(x)
    return y
X = [[12,7,3],[4 ,5,6],[7 ,8,9]]

print('Inverse Matrix of X :',inversematrix(X))