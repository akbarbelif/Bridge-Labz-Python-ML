#sudo apt-get install python-pip python3-pip
#sudo pip3 install -U numpy
def mulmatrix(x,y):
    print("Matrix X :",x)
    print("Matrix Y :",y)
    #Matrix multiplication
    result=[[x[i][j] * y[j] for j in range(len(x[0]))] for i in range(len(x))]
    return result

X = [[ 5, 1 ,3], [ 1, 1 ,1], [ 1, 2 ,1]]
Y = [1, 2, 3]

print('Vector multiplication of X and Y :',mulmatrix(X,Y))