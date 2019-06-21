#sudo apt-get install python-pip python3-pip
#sudo pip3 install -U numpy
#Multiple from Problem 1
def mulmatrix(x,y):
    print("Matrix X :",x)
    print("Matrix Y :",y)
    #Matrix Multiply
    result=[[x[i][j]*y[i][j ] for j in range(len(x[0]))] for i in range(len(x))]
    return result

X = [[12,7,3],[4 ,5,6],[7 ,8,9]]
Y = [[5,8,1],[6,7,3],[4,5,9]]

print('Multiplication of X and Y :',mulmatrix(X,Y))