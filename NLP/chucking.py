import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer



trainspeech=state_union.raw("/home/admin1/Desktop/Deep-Learing/NLP/PM_Modi_Speech.txt")
testspeech=state_union.raw("PM_Modi_Speech.txt")

custom_speech_tokenizer=PunktSentenceTokenizer(trainspeech)

tokenized=custom_speech_tokenizer.tokenize(testspeech)

def process_chucking():
    try:

        for i in tokenized:
            words=nltk.tokenize(i)
            tagged=nltk.pos_tag(words)
            #print(tagged)
            #Create a Regualr Expression by creating a Chuck
            #We combine the part of speech tags with Regular Expression
            # + = match
            # 1 or more
            # ? = match
            # 0 or 1
            # repetitions.
            # * = match
            # 0 or MORE
            # repetitions
            # .= Any character except a new line

            chuckGram= r"""Chuck: {<RB.?>*<VB.?><NNP>+<NN>?}"""
            chuck=nltk.RegexpChunkParser(chuckGram)
            print(chuck)
        pass

    except Exception as e :

        print(str(e))
        pass


