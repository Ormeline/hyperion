import spacy
import numpy as np

nlp = spacy.load("en_core_web_md")

# Define a function to return the most similar movie based on a given description
def get_similar_movie(description):
    similarities = []
    sentence_to_compare = nlp(description)
    for movie in movies:
        movie_description = movie.split(":")[1].strip()
        similarities.append(sentence_to_compare.similarity(nlp(movie_description)))
    index = np.argmax(similarities)
    return movies[index].split(":")[0].strip()

# Read in the movies.txt file. Each separate line is a description of a different movie
with open("movies.txt", "r") as file:
    movies = file.readlines()

# Get the user's recommended movie based on the description of "Planet Hulk"
description = "Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk land on the planet Sakaar where he is sold into slavery and trained as a gladiator."
recommended_movie = get_similar_movie(description)

# Print the recommended movie to the user
print(f"The movie you should watch next is: {recommended_movie}")