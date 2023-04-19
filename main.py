import nltk
import numpy as np
import pandas as pd
import random
import string  # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')  # first-time use only nltk.download ('wordnet') # first-time use only
nltk.download('wordnet')
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer

f = pd.read_csv('movies.csv')

# raw = f.read()
# raw = raw.lower()  # converts to lowercase
for _, row in f.iterrows():
    movie_title = row['name']
    movie_director = row['director']
    movie_cast = row['star']
    movie_genre = row['genre']
    movie_year = row['year']
# sent_tokens = nltk.sent_tokenize(f)  # converts to list of sentences
# word_tokens = nltk.word_tokenize(f)  # converts to list of words

print(movie_title[:2])
# print(word_tokens[:2])

lemmer = nltk.stem.WordNetLemmatizer()


# WordNet is a semantic y-oriented dictionary of English included In NK.
def lemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(word):
    return lemTokens(nltk.word_tokenize(word.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey" "*nods", "hi there", "hello", "I am glad! you are talking to me"]


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)



def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response


flag = True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while flag:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("ROBO: You are welcome..")
        else:
            if greeting(user_response) is not None:
                print("ROBO: " + greeting(user_response))
            else:
                print("ROBO: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("ROBO: Bye! take care..")
