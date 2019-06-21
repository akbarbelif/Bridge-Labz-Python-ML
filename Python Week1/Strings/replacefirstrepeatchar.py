def replaceword():
    myword="Restart"
    result=""
    print(myword)
    for i in range(1,len(myword)):
        if myword[i].lower() == myword[0].lower():
            result=result+'$'
        else:
            result=result + myword[i]
        
        #my_dict.update(dict([(i,myword)])
    return result

print('Replace first with $: ',replaceword())