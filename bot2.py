import nltk
import numpy as np
import pandas as pd
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Install punkt and wordnet if not downloaded
nltk.download('punkt')
nltk.download('wordnet')

# Read in the fitness file
with open('fitness.txt', 'r', errors='ignore') as f:
    raw = f.read().lower()

# Tokenize the raw text into sentences and words
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

# Define a lemmatizer to reduce words to their root form
lemmer = nltk.stem.WordNetLemmatizer()


def lemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


# Remove punctuation from input string
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)


# Normalize user input
def LemNormalize(text):
    return lemTokens(nltk.word_tokenize(remove_punctuation(text).lower()))


# Define a list of greeting inputs and corresponding responses
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me."]


# Define the greeting function to return a random greeting response if the input is a greeting
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Define the response function
def response(user_response, q_table, state):
    robo_response = ''

    # Add user response to sentence tokens
    sent_tokens.append(user_response)

    # Vectorize sentence tokens using TF-IDF
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)

    # Compute cosine similarity between user input and sentence tokens
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    # If the cosine similarity is 0, return an "I don't understand" response
    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response, state
    else:
        # Retrieve the most similar sentence token and remove user response from sentence tokens
        robo_response = robo_response + sent_tokens[idx]
        sent_tokens.remove(user_response)

        # Update the state based on the robo response
        state = LemNormalize(robo_response)

        # Choose the action based on the highest Q-value for the current state
        action = np.argmax(q_table.loc[state, :])

        # If the Q-value for the action is 0, return the robo response
        if q_table.loc[state, action] == 0:
            return robo_response, state

        # Otherwise, choose a random action with probability epsilon
        else:
            epsilon = 0.2
            if np.random.uniform() < epsilon:
                action = np.random.choice(q_table.columns)

            # Return the robo response and action
            return robo_response, action


# Define the Q-learning function
def q_learning(agent, state, action, reward, next_state, next_action, alpha, gamma):
    """
    Update the Q-value of a given state-action pair using Q-learning algorithm
    """
    q_val = agent.get_q_value(state, action)
    next_q_val = agent.get_q_value(next_state, next_action)
    new_q_val = q_val + alpha * (reward + gamma * next_q_val - q_val)
    agent.set_q_value(state, action, new_q_val)
