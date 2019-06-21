def cointoss_H(x,y,z):
    print("Coin 1 :",x)
    print("Coin 2 :",y)
    print("Coin 3 :",z)
    #Matrix Multiply
    mytuple=tuple() 
    counts=0          
    mytuple=[x[i]+y[j]+z[k] for k in range(0,2) for j in range(0,int(2)) for i in range(0,2)]
    #mytuple=result
    print(mytuple)
    a=input('Enter the pattern :')
    for q in range(len(mytuple)):
        for w in mytuple[q]:
            for r in range(len(a)):
                if w ==a[r].upper():
                    counts+=1
                    break
            
    print(a.upper()+" Pattern :",counts)
    return float(counts)/len(mytuple),a.upper()

X = ['H','T']
Y = ['H','T']
Z = ['H','T']

print('Probabity of pattern coin tons:',cointoss_H(X,Y,Z))