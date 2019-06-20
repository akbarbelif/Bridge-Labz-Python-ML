from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps=PorterStemmer()
Example_sent="It very Important for python coder to code properly,while you are coding pythonly coding should be persis and accurate."

words=word_tokenize(Example_sent)

print("Tokenize Word:\n",words)
count=0
for w in words:
    count +=1
    print(count,ps.stem(w))

