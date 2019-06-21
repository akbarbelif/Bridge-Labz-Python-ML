def smallestnum():
    smallest_num=list()
    val=input('Enter number :')
    smallest_num=val.split(",")
    temp=0
    for i in smallest_num:
        if temp < int(i):
            for j in smallest_num:
                if int(i) < int(j):
                    temp = int(i)

    print("Smallest number in List :",temp)

smallestnum()


        
        
