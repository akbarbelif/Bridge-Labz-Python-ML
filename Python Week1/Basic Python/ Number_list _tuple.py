#Lists and Tuples store one or more objects or values in a specific order
def switch_case():
    s=input('Enter 1: List and 2: Tuples = ')
    switch_demo(s)


def list_type():
    V= input('Enter Sets of Numbers: ')
    l=list(V.split(","))
    print(l)

def tuples_type():
    V=input('Enter Sets of Numbers: ')
    t=tuple(V.split(","))
    print(t)

def default():
    print("Default")

def Input_num():
    I=input('Enter Sets of Numbers: ')
    return I

def switch_demo(argument):
     switcher = {
        1: list_type(),
        2: tuples_type(),
        default: default(),
    }
    #func = switcher.get(argument,"Invalid month")
    #print fun

switch_case()
