import spacy
from spacy.training import Example
import random

# Define the training data
TRAIN_DATA = [
    ("I love The Godfather", {"entities": [(7, 18, "MOVIE")]}),
    ("The Shawshank Redemption is my favorite movie", {"entities": [(4, 28, "MOVIE")]}),
    ("Tom Hanks is my favorite actor", {"entities": [(0, 9, "STAR")]}),
    ("I enjoy watching comedies", {"entities": [(19, 28, "GENRE")]}),
    ("The Lord of the Rings was released in 2001", {"entities": [(4, 21, "MOVIE"), (36, 40, "YEAR")]}),
    # Add more training examples here
]

# Initialize a blank spaCy model
nlp = spacy.blank("en")

# Create the entity recognizer and add it to the pipeline
# ner = nlp.create_pipe("ner")
ner = nlp.add_pipe("ner")

# Add the labels to the entity recognizer
ner.add_label("MOVIE")
ner.add_label("STAR")
ner.add_label("GENRE")
ner.add_label("YEAR")

# Train the model
optimizer = nlp.begin_training()
for i in range(20):
    # Shuffle the training data for each iteration
    random.shuffle(TRAIN_DATA)
    # Create examples and update the model
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], sgd=optimizer)
