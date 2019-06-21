def sumList():
    sum_list=list()
    val=input('Enter Number for Addition :')
    sum_list=val.split(",")
    print(sum_list)
    print(sum(int(i) for i in sum_list))

sumList()