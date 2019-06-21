#Generate Dictionary
def gen_Dict():
    n = int(input('Enter Number: '))
    d = dict()
    for x in range(1, n+1):
        d[x] = x*x
    print(d)

gen_Dict()
