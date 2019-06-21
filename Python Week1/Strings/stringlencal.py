def str_len(mylist):
    print("List: ",mylist)
    lencount=[]
    for i in mylist:
        lencount.append(len(i))
    location=lencount.index(max(lencount))
    print("Word Having max Lenght:"[0]"With a size of"[1],mylist[location],len(mylist[location]))


my_list=list()
strval=str(input('Enter Some word :'))
my_list=strval.split(",")
str_len(my_list)    