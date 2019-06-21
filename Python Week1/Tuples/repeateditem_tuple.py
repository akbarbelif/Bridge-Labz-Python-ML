def repeatedTuple(mytuple):
    rep = {x for x in mytuple if mytuple.count(x) > 1}
    return rep



print("Repeat item :",repeatedTuple(("akbar","akbar","faizan","faizan",3,3,4)))