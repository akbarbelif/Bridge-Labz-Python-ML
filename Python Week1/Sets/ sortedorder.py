def sortlist(myList):
    print("Sample List :",myList)
    return sorted(myList,key=last)

def last(n):
    return n[-1]

print("Sorted by Increasing order",sortlist([(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]))