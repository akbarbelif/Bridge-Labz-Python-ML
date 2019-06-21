def unpacktuple():
    #create tuple
    mytuple=1,2,4,5
    print("Tuples",mytuple)
    #Equal Number of variable should be define according to tuple
    t1,t2,t3,t4=mytuple
    #unpack tuple
    print("Unpack",t1+t2+t3+t4)

unpacktuple()