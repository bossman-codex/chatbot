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
    for _, row in movie_data.sample(frac=1).iterrows():  # shuffle the rows using `sample()` and loop through them
        movie_title = row['name']
        movie_director = row['director']
        movie_star = str(row['star'])
        movie_genre = row['genre']
        movie_year = str(row['year'])
        sample_text.append("Here is something on " + movie_title + ', directed by ' + movie_director + ' in ' + movie_year + ', starring ' + movie_star + ' in the ' + movie_genre + ' genre.')

    # Compute the TF-IDF vectors for the sample text
    vectorizer = TfidfVectorizer(preprocessor=preprocess)
    sample_vectors = vectorizer.fit_transform(sample_text)

    # Define a function to find the best response for a given input text
    def find_best_response(inputField):
        # Compute the TF-IDF vector for the input text
        input_vector = vectorizer.transform([inputField])

        # Compute the cosine similarities between the input vector and the sample vectors
        similarities = cosine_similarity(input_vector, sample_vectors)[0]

        print(similarities)
        # Find the index of the sample vector with the highest cosine similarity
        best_index = np.argmax(similarities)

        title = movie_data.loc[best_index, 'name']
        star = str(movie_data.loc[best_index, 'star'])
        genre = movie_data.loc[best_index, 'genre']
        year = str(movie_data.loc[best_index, 'year'])
        return title, star, genre, year

    # Start the chat loop
    print('Hello! I am a movie chatbot. You can ask me about any topic and I will recommend a movie that is related to it. If you want to quit, just say "quit".')
    while True:
        input_text = input('> ')
        if input_text.lower() in ['quit', 'exit']:
            print('Goodbye!')
            break
        elif input_text.lower() in ['hi', 'hello', 'hey']:
            print('Hello! How can I help you?')
        else:
            # Preprocess the input text and find the best matching movie title
            input_text = preprocess(input_text)
            movie_title, movie_star, movie_genre, movie_year = find_best_response(input_text)
            print(f"I recommend {movie_title}, directed by {movie_data.loc[movie_data['name'] == movie_title, 'director'].iloc[0]} in {movie_year}, starring {movie_star} in the {movie_genre} genre.")

chatbot()