#sudo apt-get install python-pip python3-pip
#sudo pip3 install -U numpy
def mulmatrix(x,y):
    print("Matrix X :",x)
    print("Matrix Y :",y)
    #Matrix multiplication
    result=[[x[i][j] * y for j in range(len(x[0]))] for i in range(len(x))]
    return result

X = [[12,7,3],[4 ,5,6],[7 ,8,9]]
Y = 9

print('Scalar multiplication of X and Y :',mulmatrix(X,Y))