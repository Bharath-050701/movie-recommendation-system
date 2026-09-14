import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load movie dataset
movies = pd.read_csv("movies.csv")


# Convert movie genres into numerical vectors
vectorizer = TfidfVectorizer()

movie_vectors = vectorizer.fit_transform(movies["genre"])


# Calculate similarity between all movies
similarity = cosine_similarity(movie_vectors)


# Recommendation function
def recommend_movies(movie_name):

    movie_name = movie_name.lower()

    # Find the movie
    matches = movies[movies["title"].str.lower() == movie_name]

    # If movie doesn't exist
    if matches.empty:
        return []

    # Get movie's index
    movie_index = matches.index[0]

    # Get similarity scores
    similarity_scores = list(
        enumerate(similarity[movie_index])
    )

    # Sort from highest similarity to lowest
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Store recommendations
    recommendations = []

    # Get top 5 similar movies
    for index, score in similarity_scores[1:6]:

        recommendations.append(
            movies.iloc[index]["title"]
        )

    return recommendations
    