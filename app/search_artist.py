import csv
import difflib
from pathlib import Path


def load_csv(file_path: Path, delimiter='\t'):
    """
    Loads a CSV file (or .dat file) and returns a list of dictionaries.
    Assumes the file has a header row.
    """
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
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
    mapping_path = Path("artist_mapping2.dat")
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