import nltk
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Define a function to preprocess the text
def preprocess(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]

    return ' '.join(tokens)


# Define the chatbot
def chatbot():
    # Read the data from the movie CSV file using pandas and create a list of sample text
    movie_data = pd.read_csv('movies.csv')
    sample_text = []
    for _, row in movie_data.iterrows():
        movie_title = row['name']
        movie_director = row['director']
        movie_cast = str(row['star'])
        movie_genre = row['genre']
        movie_year = str(row['year'])
        sample_text.append(movie_title + ' ' + movie_director + ' ' + movie_cast + ' ' + movie_genre + ' ' + movie_year)

    # Compute the TF-IDF vectors for the sample text
    vectorizer = TfidfVectorizer(preprocessor=preprocess)
    sample_vectors = vectorizer.fit_transform(sample_text)

    # Define a function to find the best response for a given input text
    def find_best_response(input_text):
        # Compute the TF-IDF vector for the input text
        input_vector = vectorizer.transform([input_text])

        # Compute the cosine similarities between the input vector and the sample vectors
        similarities = cosine_similarity(input_vector, sample_vectors)[0]

        # Find the index of the sample vector with the highest cosine similarity
        best_index = np.argmax(similarities)

        # Return the corresponding movie information
        title = movie_data.loc[best_index, 'name']
        star = str(movie_data.loc[best_index, 'star'])
        genre = movie_data.loc[best_index, 'genre']
        year = str(movie_data.loc[best_index, 'year'])
        return title, star, genre, year

    decision_tree = {
        'title': {'cast': 'The recommended movie is {0}. Would you like to know more about the cast?',
                  'genre': 'The recommended movie is {0}. Would you like to know more about the genre?',
                  'year': 'The recommended movie is {0}. Would you like to know more about the year?'},
        'cast': {'title': 'The recommended cast is {0}. Would you like to know more about the movie title?',
                 'genre': 'The recommended cast is {0}. Would you like to know more about the genre?',
                 'year': 'The recommended cast is {0}. Would you like to know more about the year?'},
        'genre': {'title': 'The recommended genre is {0}. Would you like to know more about the movie title?',
                  'cast': 'The recommended genre is {0}. Would you like to know more about the cast?',
                  'year': 'The recommended genre is {0}. Would you like to know more about the year?'},
        'year': {'title': 'The recommended year is {0}. Would you like to know more about the movie title?',
                 'cast': 'The recommended genre is {0}. Would you like to know more about the cast?',
                 'genre': 'The recommended cast is {0}. Would you like to know more about the genre?'}
    }

    # Define a function to determine the chatbot's response based on the decision tree
    def get_chatbot_response(input_text, context):
        if context == 'title':
            if 'cast' in input_text:
                return decision_tree['title']['cast'].format(find_best_response(input_text)[1]), 'cast'
            elif 'genre' in input_text:
                return decision_tree['title']['genre'].format(find_best_response(input_text)[2]), 'genre'
            elif 'year' in input_text:
                return decision_tree['title']['year'].format(find_best_response(input_text)[3]), 'year'
            else:
                return "I'm sorry, I didn't understand. Please try again.", context
        elif context == 'cast':
            if 'title' in input_text:
                return decision_tree['cast']['title'].format(find_best_response(input_text)[0]), 'title'
            elif 'genre' in input_text:
                return decision_tree['cast']['genre'].format(find_best_response(input_text)[2]), 'genre'
            elif 'year' in input_text:
                return decision_tree['cast']['year'].format(find_best_response(input_text)[3]), 'year'
            # else:
            #     return "I'm sorry, I didn't understand. Please try again.", context

# Start the chat loop

    context = 'title'

    print(
        'Hello! I am a movie chatbot. You can ask me about any topic and I will recommend a movie that is related to it. If you want to quit, just say "quit".')
    while True:
        input_text = input('> ')
        if input_text.lower() in ['quit', 'exit']:
            print('Goodbye!')
            break
        elif input_text.lower() in ['hi', 'hello', 'hey']:
            print('Hello! How can I help you?')
        else:
            # Preprocess the input text and find the best matching movie title
           # input_text = preprocess(input_text)
            # response = get_chatbot_response(input_text,)
            # print(response)
            # Determine the chatbot's response based on the input text and context
            response, new_context = get_chatbot_response(input_text, "title")

            # Update the chat context
            context = new_context

            # Output the chatbot's response
            print(response)

chatbot()