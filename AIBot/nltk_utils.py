import nltk
import numpy as np
#region NLP Pipeline

#pretrained tokenizer indirmemizi sağlıyor yalnızca ilk çalıştırdığımız zaman indirmemiz yeterli
#nltk.download("punkt")

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def Tokenize(sentence):
    return nltk.word_tokenize(sentence)

def Stemming(word):
    return stemmer.stem(word.lower())

def BagOfWords(tokenized_sentence,all_words):
    tokenized_sentence = [Stemming(word) for word in tokenized_sentence]

    bag = np.zeros(len(all_words),dtype=np.float32)
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1.0
    return bag
       
#endregion

#region Deneme


'''
sentence = ["hello","how","are","you"]
words = ["hi","hello","I","you","bye","thank","cool"]
bag = BagOfWords(sentence,words)
print(bag)

sentence = "How are you my annoying friend?"

tokenizedSentence = Tokenize(sentence)

print(sentence)
print(tokenizedSentence)

stemArray = []
for word in tokenizedSentence:
    stem = Stemming(word)
    stemArray.append(stem)

print(stemArray)
'''
#endregion