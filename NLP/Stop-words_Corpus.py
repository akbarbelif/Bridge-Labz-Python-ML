from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."

stop_words=set(stopwords.words("english"))
print("Set of Curpus English Words,already perdefine in the library:\n",stop_words)

word=word_tokenize(EXAMPLE_TEXT)

filter_Nonstopword=[]
# #
# # for w in word:
# #     if w not in stop_words:
# #         listed_Nonstopword.append(w)
# #
#
filter_Nonstopword=[w for w in word if w not in stop_words]
#
print("\nFollowing words doesnt Much in stopwords:\n",filter_Nonstopword)



