def count_matchlst(myList):
    count=0
    print("List :",myList)
    for word in myList:
        if len(word)>1 and word[0]==word[-1]:
            count +=1
    return count

print("Match Found :",count_matchlst(['abc', 'xyz', 'aba', '1221']))