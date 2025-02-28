import logging
from pathlib import Path
from typing import Tuple, List, Dict
import pickle
import implicit
import scipy
import scipy.sparse as sp
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request
from app.artist_mapping.data import load_user_artists, ArtistRetriever
import app.search_artist as search_artist_lib

# Blueprint setup
main = Blueprint('main', __name__)


class ImplicitRecommender:
    """The ImplicitRecommender class computes recommendations for a given user
    using the implicit library.

    Attributes:
        - artist_retriever: an ArtistRetriever instance
        - implicit_model: an implicit model
        - user_artists: the user-artists matrix
        - genre_model: ALS model for genre recommendations
        - genre_matrix: user-genre matrix
    """

    def __init__(
            self,
            artist_retriever: ArtistRetriever,
            implicit_model: implicit.recommender_base.RecommenderBase,
            user_artists: sp.csr_matrix,
            genre_model: implicit.als.AlternatingLeastSquares = None,
            genre_matrix: sp.csr_matrix = None,
            genre_to_index: dict = None
    ):
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model
        self.user_artists = user_artists
        self.genre_model = genre_model
        self.genre_matrix = genre_matrix
        self.genre_to_index = genre_to_index

    def fit(self) -> None:
        """Fit the ALS model to the user-artist matrix."""
        self.implicit_model.fit(self.user_artists)

    def recommend_by_genre(self, genre: str, n: int = 10) -> Tuple[List[str], List[float]]:
        """Return the top n recommended artists for a given genre."""
        if genre not in self.genre_to_index:
            return [], []

        genre_index = self.genre_to_index[genre]
        artist_ids, scores = self.genre_model.recommend(genre_index, self.genre_matrix[genre_index], N=n)
        artists = [self.artist_retriever.get_artist_name_from_id(artist_id) for artist_id in artist_ids if
                   artist_id in self.artist_retriever._artists_df.index]
        return artists, scores

    def recommend_by_artist_list(
            self, artist_id_list: list, n: int = 10
    ) -> Tuple[List[str], List[float]]:
        """Return the top n recommendations for the given artist list."""

        filter_values = np.array(artist_id_list, dtype=np.int32)
        mask = np.isin(self.user_artists.indices, filter_values)

        filtered_rows = self.user_artists.nonzero()[0][mask]
        filtered_cols = self.user_artists.indices[mask]
        filtered_data = self.user_artists.data[mask]

        groupped_matrix = scipy.sparse.csr_matrix(
            (filtered_data, (filtered_rows, filtered_cols))
        )

        row_sums = groupped_matrix.sum(axis=1)  # Row-wise sum
        row_sums = np.array(row_sums).flatten()

        sorted_indices = np.argsort(row_sums)[::-1]
        top_row_index = sorted_indices[0]

        artist_ids, scores = self.implicit_model.recommend(
            top_row_index, self.user_artists[top_row_index], N=n
        )
        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]
        return artists, scores

# artist based recommendation

def recommend_based_on_artist(artist_name):
    pickle_path = Path("recommender.pkl")

    if pickle_path.exists():
        # Load model from pickle
        with open(pickle_path, 'rb') as file:
            recommender = pickle.load(file)
        print("Model loaded from file.")
    else:
        # Load data
        user_artists = load_user_artists(Path("data/lastfmdata/user_artists.dat"))
        artist_retriever = ArtistRetriever()
        artist_retriever.load_artists(Path("data/lastfmdata/artists.dat"))

        # Instantiate ALS using implicit
        implicit_model = implicit.als.AlternatingLeastSquares(
            factors=50, iterations=10, regularization=0.01, random_state=1907
        )

        # Create recommender and fit the model
        recommender = ImplicitRecommender(artist_retriever, implicit_model, user_artists)
        recommender.fit()

        # Save to pickle
        with open(pickle_path, 'wb') as file:
            pickle.dump(recommender, file)
        print("Model created and saved to file.")

    artist_id = recommender.artist_retriever.get_artist_id_from_name(artist_name)
    print(artist_id)
    # Return the recommendations instead of just printing
    return recommender.recommend_by_artist_list([artist_id], n=10)

def find_match(search_term):
    """
    Uses the search_artist function (from search_artist.py) to find
    the best matching artist record based on the search_term.
    It prints the details of the match and returns the record.
    """
    result = search_artist_lib.return_best_match(search_term)
    if result:
        print("Best match found from {}:".format(result.get("source")))
        for key, value in result.items():
            print(f"{key}: {value}")
        return result
    else:
        print("No match found.")
        return None


def recommend_based_on_search(search_term) -> Dict[str, any]:
    """
    Given a search term, this function:
      1. Finds the best matching artist record using find_match.
      2. Extracts the artist name from the result.
      3. Uses the existing recommend_based_on_artist function to get recommendations.
      4. Returns the matched artist name along with the recommendations.
    """
    match = find_match(search_term)
    if not match:
        return None

    # Use the appropriate field for the artist name.
    artist_name = match.get("lastfm_artist_name") or match.get("name")
    if not artist_name:
        print("No valid artist name found in the matched record.")
        return None

    # Get recommendations
    artists, scores = recommend_based_on_artist(artist_name)

    return jsonify({"matched_artist": artist_name, "artists": artists, "scores": scores.tolist()})

# genre based recommendation

def create_user_genre_matrix():
    """Create a user-genre interaction matrix from Last.fm dataset."""
    df_user_artists = pd.read_csv("data/lastfmdata/user_artists.dat", sep="\t", encoding="latin1")
    df_tags = pd.read_csv("data/lastfmdata/user_taggedartists.dat", sep="\t", encoding="latin1")
    df_tag_names = pd.read_csv("data/lastfmdata/tags.dat", sep="\t", encoding="latin1")

    df_tags = df_tags.merge(df_tag_names, on='tagID', how='left')
    df_user_genre = df_user_artists.merge(df_tags, on="artistID", how="inner")

    if "userID_x" in df_user_genre.columns:
        df_user_genre.rename(columns={"userID_x": "userID"}, inplace=True)
    if "userID_y" in df_user_genre.columns:
        df_user_genre.drop(columns=["userID_y"], inplace=True)

    df_user_genre = df_user_genre.groupby(["userID", "tagValue"])['weight'].sum().reset_index()

    user_to_index = {user: i for i, user in enumerate(df_user_genre["userID"].unique())}
    genre_to_index = {genre: i for i, genre in enumerate(df_user_genre["tagValue"].unique())}

    rows = df_user_genre["userID"].map(user_to_index)
    cols = df_user_genre["tagValue"].map(genre_to_index)
    data = df_user_genre["weight"]

    user_genre_matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(user_to_index), len(genre_to_index)))
    return user_genre_matrix, genre_to_index


def train_genre_model():
    """Train ALS model for genre recommendations."""
    genre_matrix, genre_to_index = create_user_genre_matrix()
    als_genre_model = implicit.als.AlternatingLeastSquares(factors=50, iterations=10, regularization=0.01)
    als_genre_model.fit(genre_matrix)

    with open("als_genre_model.pkl", "wb") as file:
        pickle.dump((als_genre_model, genre_matrix, genre_to_index), file)

    return als_genre_model, genre_matrix, genre_to_index


def recommend_based_on_genre(genre_names: List[str], n: int = 10, exclude_ids: set = None):
    """Recommend artists based on multiple genres, excluding specific artist IDs and ensuring uniqueness."""

    pickle_path = Path("als_genre_model.pkl")
    if not pickle_path.exists():
        train_genre_model()

    with open(pickle_path, "rb") as file:
        genre_model, genre_matrix, genre_to_index = pickle.load(file)

    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("data/lastfmdata/artists.dat"))

    artist_mapping_path = Path("data/lastfmdata/artist_mapping_2.dat")
    if artist_mapping_path.exists():
        artist_mapping = pd.read_csv(artist_mapping_path, sep="\t", encoding="latin1")
    else:
        artist_mapping = pd.DataFrame(columns=["spotify_artist_name", "spotify_artist_id"])

    matched_genres = []
    recommended_artists = set()  # Use a set to avoid duplicates
    artist_id_mapping = {}

    for genre_name in genre_names:
        if genre_name not in genre_to_index:
            continue  # Just skip if genre is not found

        genre_index = genre_to_index[genre_name]

        if genre_index >= genre_matrix.shape[0]:
            continue  # Skip if index is out of range

        try:
            artist_ids, scores = genre_model.recommend(genre_index, genre_matrix[genre_index], N=n)
        except IndexError:
            continue  # Skip if recommendation fails

        matched_genres.append(genre_name)

        for artist_id in artist_ids:
            if artist_id in artist_retriever._artists_df.index:
                artist_name = artist_retriever.get_artist_name_from_id(artist_id)
                if artist_name and (exclude_ids is None or str(artist_id) not in exclude_ids):
                    recommended_artists.add(artist_name)
                    spotify_id = artist_retriever.get_spotify_artist_id_from_name(artist_name, artist_mapping)
                    artist_id_mapping[artist_name] = spotify_id if spotify_id else None

    return {
        "matched_genres": matched_genres,
        "recommended_artists": list(recommended_artists),  # Convert set to list
        "artist_ids": artist_id_mapping
    }






