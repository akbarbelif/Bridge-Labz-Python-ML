def cointoss_HHH(x,y,z):
    print("Coin 1 :",x)
    print("Coin 2 :",y)
    print("Coin 3 :",z)
    mytuple=tuple() 
    counts=0          
    mytuple=[x[i]+y[j]+z[k] for k in range(0,2) for j in range(0,2) for i in range(0,2)]
    #mytuple=result
    print(mytuple)
    a=input('Enter the pattern :')
    counts=mytuple.count(a.upper())
    return float(counts)/len(mytuple),a.upper()

X = ['H','T']
Y = ['H','T']
Z = ['H','T']

print('Probabity of pattern coin tons:',cointoss_HHH(X,Y,Z))