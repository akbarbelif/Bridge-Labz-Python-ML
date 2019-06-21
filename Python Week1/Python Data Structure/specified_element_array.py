from array import *

def spec_arr():
    array=[1,2,2,3,6,5,8]
    val=int(input('Enter Specified Element: '))
    count=0
    for i in array:
        if i == val :
            count += 1
    
    print("Number of occurrences", count)

spec_arr()