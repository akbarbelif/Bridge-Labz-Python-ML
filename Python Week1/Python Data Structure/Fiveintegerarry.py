#Array Index Search
from array import *

def arr():
    m=0
    array=[1,2,3,4,5]
    for i in array:
        if m > i:
            print('Input Value Missing.....')
    #Input Index Number for search
    m=int(input('Enter Index Number: '))
    mi=array.index(m)
    print (mi)
 
arr()
