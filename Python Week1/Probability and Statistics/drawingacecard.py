def acecard():
    #Total number of Ace card inside a dock 4
    s=0
    tries=0
    for prob in range(1,53):
        if prob < 5:
            s+=1
        tries +=1
    print(tries)
    print('Probabity of getting ace after 52 tries',float(s)/tries)
acecard()