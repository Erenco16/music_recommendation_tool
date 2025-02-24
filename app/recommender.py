"""This module features the ImplicitRecommender class that performs
recommendation using the implicit library.
"""


from pathlib import Path
from typing import Tuple, List
import pickle

import implicit
import scipy
import numpy as np

from app.artist_mapping.data import load_user_artists, ArtistRetriever

import search_artist as search_artist_lib

class ImplicitRecommender:
    """The ImplicitRecommender class computes recommendations for a given user
    using the implicit library.

    Attributes:
        - artist_retriever: an ArtistRetriever instance
        - implicit_model: an implicit model
        - user_artists: the user-artists matrix
    """

    def __init__(
        self,
        artist_retriever: ArtistRetriever,
        implicit_model: implicit.recommender_base.RecommenderBase,
        user_artists: scipy.sparse.csr_matrix,
    ):
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model
        self.user_artists = user_artists

    def fit(self) -> None:
        """Fit the model to the user artists matrix."""
        self.implicit_model.fit(self.user_artists)

    def recommend(self, user_id: int, n: int = 10) -> Tuple[List[str], List[float]]:
        """Return the top n recommendations for the given user."""
        artist_ids, scores = self.implicit_model.recommend(
            user_id, self.user_artists[user_id], N=n
        )
        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]
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

def recommend_based_on_search(search_term):
    """
    Given a search term, this function:
      1. Finds the best matching artist record using find_match.
      2. Extracts the artist name from the result.
      3. Uses the existing recommend_based_on_artist function to get recommendations.
    """
    match = find_match(search_term)
    if not match:
        return None

    # Use the appropriate field for the artist name.
    # If the record comes from artist_mapping2.dat, we expect a 'lastfm_artist_name';
    # Otherwise, fall back to the 'name' field.
    artist_name = match.get("lastfm_artist_name") or match.get("name")
    if not artist_name:
        print("No valid artist name found in the matched record.")
        return None

    # Return recommendations based on the best match.
    return recommend_based_on_artist(artist_name)