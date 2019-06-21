def coloncopy():
    #create tuple
    mytuple=([],"hello","world")
    print(mytuple) 
    #copy tuple at 1st postion
    mytuple[0].append("hi")
    print(mytuple)
coloncopy()