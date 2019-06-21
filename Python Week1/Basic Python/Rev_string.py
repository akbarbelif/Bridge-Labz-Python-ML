#Enter First_Name
def FirstName():
    s=True
    First=input('What is your FirstName ')
    if not First:
        s=False
    return First
#Enter Last_Name
def LastName():
    s=True
    Last=input('What is your LastName ')
    if not Last:
        s=False
    return Last


#Reverse_Name
def Revname(Rev_string):
    Rev= str(Rev_string[::-1])
    newtxt=" "
    for t in Rev:
        newtxt += " " + t
    return newtxt

F=str(FirstName())
L=str(LastName())
R=str(Revname(F+L))
print(R)