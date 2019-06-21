#Define and declare dictionary 
def dec_sort():
    #y={'carl':40,'alan':2,'bob':1,'danny':3} 
    y={}
    y[0]='carl'
    y[1]='alan'
    y[2]='bob'
    y[3]='danny'
    
    print(y)
    #Convert the given Dictionary into List
    l=list(y.items())
    l.sort()
    print()
    print('Ascending Order ',l)
    l=list(y.items())
    l.sort(reverse=True)
    print('Descending Order ',l)
    #Convert the given List into Dictionary
    dic=dict(l)
    print("Dictionary",dic)
    print()

dec_sort()