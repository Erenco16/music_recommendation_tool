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

    # Merge genre tags with tag names
    df_tags = df_tags.merge(df_tag_names, on='tagID', how='left')

    # Merge user-artist interaction data with genre tags
    df_user_genre = df_user_artists.merge(df_tags, on="artistID", how="inner")

    # Fix column naming issues
    if "userID_x" in df_user_genre.columns:
        df_user_genre.rename(columns={"userID_x": "userID"}, inplace=True)
    if "userID_y" in df_user_genre.columns:
        df_user_genre.drop(columns=["userID_y"], inplace=True)

    # Sum interactions per user-genre pair
    df_user_genre = df_user_genre.groupby(["userID", "tagValue"])['weight'].sum().reset_index()

    # Map users and genres to matrix indices
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

    # Train ALS model
    als_genre_model = implicit.als.AlternatingLeastSquares(
        factors=100,  # Increase factors for better embedding
        iterations=15,  # More iterations for better convergence
        regularization=0.05  # Adjusted to prevent overfitting
    )
    als_genre_model.fit(genre_matrix)

    # Save model
    with open("als_genre_model.pkl", "wb") as file:
        pickle.dump((als_genre_model, genre_matrix, genre_to_index), file)

    return als_genre_model, genre_matrix, genre_to_index


def get_artists_by_genre(tagIDs, tagged_artists_df, n=10, exclude_ids=None):
    """Fetch top artists for the given genre IDs."""

    if not isinstance(tagged_artists_df, pd.DataFrame):
        raise ValueError("Expected tagged_artists_df to be a DataFrame, but got {}".format(type(tagged_artists_df)))

    if "tagID" not in tagged_artists_df.columns or "artistID" not in tagged_artists_df.columns:
        raise KeyError("Missing required columns 'tagID' or 'artistID' in tagged_artists_df")

    if not isinstance(tagIDs, list):
        print(f"⚠️ Expected tagIDs to be a list, but got {type(tagIDs)}. Converting to list.")
        tagIDs = [tagIDs]  # Convert string/int to list

    matched_artists = tagged_artists_df[tagged_artists_df["tagID"].isin(tagIDs)]

    # Exclude certain artists if needed
    if exclude_ids:
        matched_artists = matched_artists[~matched_artists["artistID"].isin(exclude_ids)]

    # Select top N artists
    return matched_artists["artistID"].value_counts().head(n).index.tolist()


def load_artist_mapping():
    """Loads the artist mapping from lastfm to Spotify, ensuring proper lookup."""
    artist_mapping_path = Path("app/artist_mapping/artist_mapping_2.dat")

    if not artist_mapping_path.exists():
        print("⚠️ Artist mapping file not found.")
        return {}

    # Load artist mapping file
    artist_mapping_df = pd.read_csv(artist_mapping_path, sep="\t", encoding="latin1")

    # Ensure all artist names are strings to avoid lookup issues
    artist_mapping_df["lastfm_artist_name"] = artist_mapping_df["lastfm_artist_name"].astype(str).str.strip()
    artist_mapping_df["spotify_artist_name"] = artist_mapping_df["spotify_artist_name"].astype(str).str.strip()
    artist_mapping_df["spotify_artist_id"] = artist_mapping_df["spotify_artist_id"].astype(str).str.strip()

    # ✅ Fix: Index by `lastfm_artist_name` so we can find Spotify IDs correctly
    return artist_mapping_df.set_index("lastfm_artist_name")["spotify_artist_id"].to_dict()


def recommend_based_on_genre(user_genres: list, n: int = 10, exclude_ids: set = None):
    """
    Recommend artists based on a given list of genres.

    :param user_genres: List of genres from the user input.
    :param n: Number of recommendations per genre.
    :param exclude_ids: Set of artist IDs to exclude (the ones user already follows).
    :return: Dictionary containing matched genres, recommended artists, and their Spotify IDs.
    """

    # Load trained ALS genre model
    pickle_path = Path("als_genre_model.pkl")
    if not pickle_path.exists():
        logging.error("ALS genre model file not found. Train it first.")
        return {"matched_genres": [], "recommended_artists": [], "artist_ids": {}}

    with open(pickle_path, "rb") as file:
        genre_model, genre_matrix, genre_to_index = pickle.load(file)

    # Load artist retriever
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("data/lastfmdata/artists.dat"))

    # Load artist mapping (Spotify ID lookup)
    artist_mapping = load_artist_mapping()

    # Load genre mapping from tags.dat
    genre_mapping = search_artist_lib.load_genre_mapping()

    # Load tagged artist data
    tagged_artists_df = pd.DataFrame(search_artist_lib.load_user_tagged_artists())
    tagged_artists_df["tagID"] = pd.to_numeric(tagged_artists_df["tagID"], errors="coerce")

    matched_genres = []
    recommended_artists = set()  # Use a set to avoid duplicates
    artist_id_mapping = {}

    for genre in user_genres:
        match = search_artist_lib.best_genre_match(genre, genre_mapping, threshold=0.8)

        if not match:
            logging.warning(f"⚠️ No match found for genre '{genre}', skipping...")
            continue  # Skip unmatched genres

        tagID = match["tagID"]
        matched_genres.append(match["genre"])  # Add matched genre name

        # Get artists tagged with this genre
        genre_artists = tagged_artists_df[tagged_artists_df["tagID"] == int(tagID)]

        if genre_artists.empty:
            logging.warning(f"⚠️ No artists found for genre '{match['genre']}' (tagID {tagID})!")
            continue  # Skip if no artists found

        # Select top N artists from the genre
        top_artist_ids = genre_artists["artistID"].value_counts().head(n).index.tolist()

        for artist_id in top_artist_ids:
            try:
                artist_id = int(artist_id)  # Ensure artist ID is an integer
                print(f"🔎 Looking up artist ID: {artist_id}")  # Debugging line

                if artist_id not in artist_retriever._artists_df.index:
                    print(f"⚠️ Skipping artist ID {artist_id}: Not found in dataset!")
                    continue  # Skip missing artists

                artist_name = artist_retriever.get_artist_name_from_id(artist_id)
                print(f"✅ Found artist: {artist_name} for ID {artist_id}")  # Debugging line

                if not artist_name:
                    print(f"⚠️ No name found for artist ID {artist_id}, skipping...")
                    continue  # Skip if no name found

                # Exclude artists user already follows
                if exclude_ids and artist_mapping.get(artist_name) in exclude_ids:
                    print(f"⚠️ Excluding artist {artist_name} (ID {artist_id}) because user already follows.")
                    continue

                recommended_artists.add(artist_name)  # Add to recommendation set
                artist_id_mapping[artist_name] = artist_mapping.get(artist_name, "Unknown ID")

            except Exception as e:
                logging.error(f"⚠️ Error processing artist ID {artist_id}: {e}")
                continue  # Skip artist on error

    return {
        "matched_genres": matched_genres,
        "recommended_artists": list(recommended_artists),
        "artist_ids": artist_id_mapping
    }


def load_artist_metadata():
    """Load the artist metadata from a CSV file, ensuring it is a DataFrame."""

    file_path = "data/lastfmdata/artists.dat"

    try:
        artists_df = pd.read_csv(file_path, sep="\t", encoding="latin1")

        # **✅ Ensure `artists_df` is a DataFrame**
        if not isinstance(artists_df, pd.DataFrame):
            raise ValueError(f"⚠️ Expected artists_df to be a DataFrame, but got {type(artists_df)}")

        if "id" not in artists_df.columns or "name" not in artists_df.columns:
            raise KeyError("⚠️ Missing required columns 'id' or 'name' in artists.dat")

        artist_mapping = dict(zip(artists_df["id"], artists_df["name"]))  # {artistID: artistName}
        logging.info(f"✅ Artist metadata loaded successfully with {len(artist_mapping)} entries.")

        return artist_mapping

    except FileNotFoundError:
        logging.error(f"⚠️ Artist metadata file not found at {file_path}")
        return {}

    except Exception as e:
        logging.error(f"⚠️ Error loading artist metadata: {e}")
        return {}