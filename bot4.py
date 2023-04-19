import nltk
import numpy as np
import pandas as pd
import random
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords

nltk.download('punkt')  # first-time use only nltk.download ('wordnet') # first-time use only
nltk.download('wordnet')


# Define a function to preprocess the text
def preprocess(text):
    # Replace hyphens with spaces
    text = text.replace('-', '')

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]

    return ' '.join(tokens)


# Define a function to extract the main words from user input
def extract_main_words(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Tag the parts of speech for each token
    pos_tags = nltk.pos_tag(tokens)

    # Keep only the nouns, adjectives, and verbs
    main_words = [token for token, pos in pos_tags if pos.startswith('N') or pos.startswith('J') or pos.startswith('V')]

    return main_words


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

    def format_movie_info(row):
        return f"{row['name']} ({row['year']}) - directed by {row['director']}, starring {row['star']} ({row['genre']})"

    sample_text = [format_movie_info(row) for _, row in movie_data.iterrows()]

    # Compute the TF-IDF vectors for the sample text
    vectorizer = TfidfVectorizer(preprocessor=preprocess)
    sample_vectors = vectorizer.fit_transform(sample_text)

    # Define a function to find the best response for a given input text
    def find_best_response(inputField):
        robo_response = ''

        # Extract the main words from the input text
        main_words = extract_main_words(inputField)

        # Create a new input text from the main words
        main_text = ' '.join(main_words)

        print(main_text)

        input_vector = vectorizer.transform([main_text])

        # Compute the cosine similarities between the input vector and the sample vectors
        similarities = cosine_similarity(input_vector, sample_vectors)[0]

        # Find the indices of the sample vectors with the highest cosine similarity
        best_indices = np.argsort(similarities)[::-1][:5]

        movie_list = []
        for index in best_indices:
            title = movie_data.loc[index, 'name']
            star = str(movie_data.loc[index, 'star'])
            genre = movie_data.loc[index, 'genre']
            year = str(movie_data.loc[index, 'year'])
            movie_list.append((title, star, genre, year))

        return movie_list

    # Start the chat loop
    print(
        'GUR: Hello! I am a movie chatbot. You can ask me about any topic and I will recommend a movie that is related to it. If you want to quit, just say "quit".')
    while True:
        input_text = input('> ')
        if input_text.lower() in ['quit', 'exit']:
            print('GUR:Goodbye!')
            break
        elif greeting(input_text.lower()) is not None:
            print("GUR: " + greeting(input_text))
        else:
            # Preprocess the input text and find the best matching movie title
            input_text = preprocess(input_text)
            movie_title = find_best_response(input_text)
            if len(movie_title) == 0:
                print("GUR: I couldn't find any matching movies.")
            else:
                print("GUR: Here are some movies that might interest you:")
                for i, movie in enumerate(movie_title, 1):
                    title, star, genre, year = movie
                    print(f"{i}. {title} ({year}) - {genre}, starring {star}")


chatbot()
