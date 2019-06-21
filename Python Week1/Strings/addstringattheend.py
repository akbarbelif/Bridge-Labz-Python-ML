def appendstring(mystring):
    result=""
    if len(mystring) < 3 :
        result_str='lenght should not be less then 2'
    else:
        for i in range(len(mystring)-3,len(mystring)):
            result = result + mystring[i]
            if result in 'ing':
                result_str=mystring+'ly'
            else:
                result_str=mystring+'ing'

    return result_str

stringval=input('Enter String :')
print('Updated String:',appendstring(stringval))