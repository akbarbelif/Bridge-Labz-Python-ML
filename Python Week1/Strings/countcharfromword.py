def countchar():
    str ="google.com"
    y = { i:str.count(i) for i in str }
    return y
print(countchar())