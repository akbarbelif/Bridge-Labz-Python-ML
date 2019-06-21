def concet_dict():
    dic1 = {1: 10, 2: 20}
    dic2 = {3: 30, 4: 40}
    dic3 = {5: 50, 6: 60}

    print("First Dictionary: ", dic1)
    print("Second Dictionary: ", dic2)
    print("Third Dictionary: ", dic3)
    #Concat Dictionary 1 with Dictionary 2
    dic1.update(dic2)
    #Concat Dictionary 1 with Dictionary 3
    dic1.update(dic3)
    print("\nAfter Concatenating two Dictionaries : ",dic1)


concet_dict()
