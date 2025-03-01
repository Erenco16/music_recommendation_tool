import csv
import difflib
from pathlib import Path
import pandas as pd

def load_csv(file_path: Path, delimiter='\t'):
    """
    Loads a CSV file and returns a list of dictionaries.
    Handles encoding issues by trying multiple encodings.
    """
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                records.append(row)
    except UnicodeDecodeError:
        print(f"⚠️ Encoding issue with {file_path}. Retrying with ISO-8859-1...")
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                records.append(row)

    return records



def best_match(search_term: str, records: list, key: str, threshold: float = 0.8):
    """
    Iterates through records comparing the search_term to the value in the specified key.
    Returns the record with the highest similarity ratio that meets the threshold.
    """
    best = None
    best_ratio = 0.0
    for rec in records:
        candidate = rec[key]
        ratio = difflib.SequenceMatcher(None, search_term.lower(), candidate.lower()).ratio()
        if ratio >= threshold and ratio > best_ratio:
            best_ratio = ratio
            best = rec
    return best, best_ratio


def return_best_match(search_term: str, threshold: float = 0.8):
    """
    Search logic that:
      1. Searches artists.dat for an artist name matching the search_term.
      2. Uses the best candidate’s name to look for a mapping in artist_mapping2.dat.
      3. If no mapping is found with the required similarity, falls back to missing_artists.dat.
    Returns the matched record along with a 'source' key indicating which file the match came from.
    """
    # Load artists.dat (assumed to be in the current directory; adjust path if needed)
    artists_path = Path("data/lastfmdata/artists.dat")
    artists = load_csv(artists_path)

    # Step 1: Find best candidate from artists.dat using the search parameter.
    candidate, cand_ratio = best_match(search_term, artists, key="name", threshold=threshold)
    if not candidate:
        print(f"No artist in {artists_path} matches '{search_term}' with at least {int(threshold * 100)}% similarity.")
        return None

    # Step 2: Look for a corresponding entry in artist_mapping2.dat using the candidate's name.
    mapping_path = Path("app/artist_mapping/artist_mapping_2.dat")
    mappings = load_csv(mapping_path)
    mapping_candidate, map_ratio = best_match(candidate["name"], mappings, key="lastfm_artist_name",
                                              threshold=threshold)
    if mapping_candidate:
        mapping_candidate["match_ratio"] = map_ratio
        mapping_candidate["source"] = "app/artist_mapping/artist_mapping_2.dat"
        return mapping_candidate
    else:
        # Step 3: Fallback search in missing_artists.dat if no mapping was found.
        missing_path = Path("app/artist_mapping/missing_artists.dat")
        missing_artists = load_csv(missing_path)
        missing_candidate, miss_ratio = best_match(candidate["name"], missing_artists, key="name", threshold=threshold)
        if missing_candidate:
            missing_candidate["match_ratio"] = miss_ratio
            missing_candidate["source"] = "app/artist_mapping/missing_artists.dat"
            return missing_candidate
        else:
            print("No relevant match found in either mapping or missing artists files.")
            return None


def best_genre_match(search_genre: str, genre_mapping: dict, threshold: float = 0.8):
    """
    Finds the closest genre match in the dataset using fuzzy matching.
    """
    genres_list = list(genre_mapping.values())  # Get genre names
    best_match = difflib.get_close_matches(search_genre.lower(), genres_list, n=1, cutoff=threshold)

    if best_match:
        matched_genre = best_match[0]
        matched_tagID = next((k for k, v in genre_mapping.items() if v == matched_genre), None)

        # Debugging print to check the mapping
        print(f"Matching Genre: {search_genre} -> {matched_genre} (tagID: {matched_tagID})")

        return {"tagID": matched_tagID, "genre": matched_genre}

    print(f"No match found for genre: {search_genre}")
    return None


def return_best_genre_match(search_genre: str, threshold: float = 0.8):
    """
    Finds the best matching genre from 'tags.dat'.
    """
    genres_path = Path("data/lastfmdata/tags.dat")
    genres_df = load_csv(genres_path)

    match = best_genre_match(search_genre, genres_df, threshold)

    if match is not None:
        return {"genre": match["tagValue"], "source": "data/lastfmdata/tags.dat"}

    print(f"No genre match found for '{search_genre}'")
    return None


def load_genre_mapping():
    """Load the genre mapping from CSV file, ensuring it's a DataFrame."""

    file_path = "data/lastfmdata/tags.dat"

    try:
        tags_df = pd.read_csv(file_path, sep="\t", encoding="latin1")

        if not isinstance(tags_df, pd.DataFrame):
            raise ValueError("Expected tags_df to be a DataFrame, but got {}".format(type(tags_df)))

        if "tagID" not in tags_df.columns or "tagValue" not in tags_df.columns:
            raise KeyError("Missing required columns 'tagID' and 'tagValue' in tags.dat")

        genre_mapping = dict(zip(tags_df["tagID"], tags_df["tagValue"]))  # {tagID: genreName}
        print("Genre mapping loaded successfully with {} entries".format(len(genre_mapping)))

        return genre_mapping

    except FileNotFoundError:
        print(f"⚠️ Genre mapping file not found at {file_path}")
        return {}

    except Exception as e:
        print(f"⚠️ Error loading genre mapping: {e}")
        return {}

def load_user_tagged_artists():
    """Loads the user-tagged artists data, mapping artistID to genreIDs."""
    tagged_artists_path = Path("data/lastfmdata/user_taggedartists.dat")
    tagged_artists_df = load_csv(tagged_artists_path)
    return tagged_artists_df

def main():
    # Load data
    tagged_artists_df = pd.DataFrame(load_user_tagged_artists())

    # Convert tagID column to numeric
    tagged_artists_df["tagID"] = pd.to_numeric(tagged_artists_df["tagID"], errors="coerce")

    for genre in ["speed metal", "black metal", "metalcore"]:
        match = best_genre_match(genre, load_genre_mapping(), threshold=0.8)

        if match:
            tagID = match["tagID"]  # Assign tagID correctly
            print(f"\n🔎 Checking Artists for Genre: {genre} (tagID: {tagID})")

            # Ensure tagID is an integer before filtering
            matched_artists = tagged_artists_df[tagged_artists_df["tagID"] == int(tagID)]

            if not matched_artists.empty:
                print(f"✅ Found {len(matched_artists)} artists tagged with {genre} (tagID {tagID})")
                print(matched_artists.head(10))  # Print first 10 matches
            else:
                print(f"⚠️ No artists found for {genre} (tagID {tagID})!")

        else:
            print(f"⚠️ No genre match found for '{genre}', skipping...")

if __name__ == "__main__":
    # Load the artist dataset
    artists_df = pd.read_csv("../data/lastfmdata/artists.dat", sep="\t", encoding="latin1")

    # Check if the artist ID exists
    test_ids = [1373, 918, 1360, 841, 938, 957]  # Replace with some IDs from your response
    for artist_id in test_ids:
        print(artists_df[artists_df["id"] == artist_id])  # Should return at least one row

