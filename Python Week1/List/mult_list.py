def mulList():
    mul_list=list()
    val=input('Enter Number for Multiplication :')
    mul_list=val.split(",")
    print(mul_list)
    tot = 1 
    for i in mul_list:
        tot *=int(i)
    print("Final Result :",tot)


mulList()