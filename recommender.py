
import pandas as pd
import ast
import requests
import os

from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("OMDB_API_KEY")

# Load TMDB datasets
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets
credits.rename(columns={"movie_id": "id"}, inplace=True)
movies = movies.merge(credits, on="id")


# Extract genres
def get_genres(data):
    try:
        return " ".join(
            [item["name"] for item in ast.literal_eval(data)]
        )
    except:
        return ""


# Extract keywords
def get_keywords(data):
    try:
        return " ".join(
            [item["name"] for item in ast.literal_eval(data)]
        )
    except:
        return ""


# Extract top 3 cast members
def get_cast(data):
    try:
        cast = ast.literal_eval(data)
        return " ".join(
            [item["name"] for item in cast[:3]]
        )
    except:
        return ""


# Extract director
def get_director(data):
    try:
        crew = ast.literal_eval(data)

        for person in crew:
            if person["job"] == "Director":
                return person["name"]

        return ""
    except:
        return ""


# Fetch movie poster using OMDb API
def get_poster(movie_name):

    if not API_KEY:
        return None

    try:
        url = "https://www.omdbapi.com/"

        params = {
            "t": movie_name,
            "apikey": API_KEY
        }

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        data = response.json()

        if data.get("Response") == "True":
            poster = data.get("Poster")

            if poster and poster != "N/A":
                return poster

    except requests.RequestException:
        pass
    except ValueError:
        pass

    return None


# Prepare movie features
movies["genres"] = movies["genres"].apply(get_genres)
movies["keywords"] = movies["keywords"].apply(get_keywords)
movies["cast"] = movies["cast"].apply(get_cast)
movies["director"] = movies["crew"].apply(get_director)


# Combine movie features
movies["tags"] = (
    movies["genres"] + " " +
    movies["keywords"] + " " +
    movies["cast"] + " " +
    movies["director"] + " " +
    movies["overview"].fillna("")
)


# Create movie vectors
cv = CountVectorizer(
    max_features=5000,
    stop_words="english"
)

vectors = cv.fit_transform(movies["tags"].fillna(""))

# Calculate cosine similarity
similarity = cosine_similarity(vectors)


# Movie recommendation function
def recommend_movies(movie_name):

    matches = movies[
        movies["title_x"].str.lower() == movie_name.strip().lower()
    ]

    if matches.empty:
        return []

    movie_index = matches.index[0]

    scores = list(enumerate(similarity[movie_index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in scores[1:6]:

        movie = movies.iloc[index]

        title = movie["title_x"]

        recommendations.append({
            "title": title,
            "poster": get_poster(title),
            "rating": round(float(movie["vote_average"]), 1),
            "year": (
                str(movie["release_date"])[:4]
                if pd.notna(movie["release_date"])
                else "N/A"
            )
        })

    return recommendations
