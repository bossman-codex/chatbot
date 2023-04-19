import nltk
import numpy as np
import pandas as pd
import random
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')  # first-time use only nltk.download ('wordnet') # first-time use only
nltk.download('wordnet')


# Define a function to preprocess the text
def preprocess(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]

    return ' '.join(tokens)


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey" "*nods", "hi there", "hello", "I am glad! you are talking to me"]


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Define the chatbot
def chatbot():
    # Read the data from the movie CSV file using pandas and create a list of sample text
    movie_data = pd.read_csv('movies.csv')
    sample_text = []
    # for _, row in movie_data.iterrows():
    #     movie_title = row['name']
    #     movie_director = row['director']
    #     movie_star = str(row['star'])
    #     movie_genre = row['genre']
    #     movie_year = str(row['year'])
    #     sample_text.append("Here is something on" + " " + movie_title + " " + "its a/an" + movie_genre + "movie")

    def format_movie_info(row):
        return f"{row['name']} ({row['year']}) - directed by {row['director']}, starring {row['star']} ({row['genre']})"

    sample_text = [format_movie_info(row) for _, row in movie_data.iterrows()]

    # Compute the TF-IDF vectors for the sample text
    vectorizer = TfidfVectorizer(preprocessor=preprocess)
    sample_vectors = vectorizer.fit_transform(sample_text)

    # Define a function to find the best response for a given input text
    def find_best_response(inputField):
        robo_response = ''
        # sample_text.append(inputField)
        # Compute the TF-IDF vector for the input text
        input_vector = vectorizer.transform([inputField])


        # Compute the cosine similarities between the input vector and the sample vectors
        similarities = cosine_similarity(input_vector, sample_vectors)[0]

        # Find the indices of the sample vectors with the highest cosine similarity
        best_indices = np.argsort(similarities)[::-1][:7]

        # Select a random index from the best indices
        random_index = random.choice(best_indices)

        # Get the movie information for the selected index
        movie_info = format_movie_info(movie_data.iloc[random_index])

        # Find the index of the sample vector with the highest cosine similarity
        #best_index = np.argmax(similarities)

        # TfidfVec = TfidfVectorizer(tokenizer=preprocess, stop_words='english')
        # tfidf = TfidfVec.fit_transform(sample_text)
        # vals = cosine_similarity(tfidf[-1], tfidf)
        # idx = vals.argsort()[0][-2]
        # flat = vals.flatten()
        # flat.sort()
        # req_tfidf = flat[-2]

        # if best_indices == 0:
        #     robo_response = robo_response + "I am sorry! I don't understand you"
        #     return robo_response
        # else:
        #     robo_response = robo_response + sample_text[idx]
        #     return robo_response

        # title = movie_data.loc[best_index, 'name']
        # star = str(movie_data.loc[best_index, 'star'])
        # genre = movie_data.loc[best_index, 'genre']
        # year = str(movie_data.loc[best_index, 'year'])
        return movie_info

    # Start the chat loop
    print(
        'Hello! I am a movie chatbot. You can ask me about any topic and I will recommend a movie that is related to it. If you want to quit, just say "quit".')
    while True:
        input_text = input('> ')
        if input_text.lower() in ['quit', 'exit']:
            print('ROBO:Goodbye!')
            break
        elif greeting(input_text.lower()) is not None:
            print("ROBO: " + greeting(input_text))
        else:
            # Preprocess the input text and find the best matching movie title
            input_text = preprocess(input_text)
            movie_title = find_best_response(input_text)
            print("ROBO:Here is something :" + " " + movie_title)
            # for role in movie_title:
            #     print(role)



chatbot()
