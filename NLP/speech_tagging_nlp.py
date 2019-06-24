import nltk
from nltk.tokenize import  word_tokenize
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
#PunktSentence Tokenizer is a unsuperier


trainspeech=state_union.raw("/home/admin1/Desktop/Deep-Learing/NLP/PM_Modi_Speech")
testspeech=state_union.raw("/home/admin1/Desktop/Deep-Learing/NLP/PM_Modi_Speech")
custom_sent_token=PunktSentenceTokenizer(trainspeech)
tokenized=custom_sent_token.tokenize(testspeech)

def process_content():
    try:
        for i in tokenized:
            words=word_tokenize(i)
            tagged=nltk.pos_tag(words)
            print(tagged)
        pass

    except Exception as e :
        print(str(e))
        pass



process_content()