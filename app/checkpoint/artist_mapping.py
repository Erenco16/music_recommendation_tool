import pandas as pd
import requests
import time
import re
import os
import Levenshtein

# File paths
MISSING_ARTISTS_FILE = "missing_artists.dat"
OUTPUT_FILE = "/data/artist_mapping.dat"

# API Variables
ACCESS_TOKEN = None
TOKEN_RETRIEVED_TIME = None
TOKEN_EXPIRY_TIME = 50 * 60  # 50 minutes

# Cache for search results
artist_cache = {}


# Function to get a new token

def get_access_token():
    global ACCESS_TOKEN, TOKEN_RETRIEVED_TIME
    if ACCESS_TOKEN and TOKEN_RETRIEVED_TIME and (time.time() - TOKEN_RETRIEVED_TIME < TOKEN_EXPIRY_TIME):
        return ACCESS_TOKEN

    request_url = "http://flask-server:5001/get-access-token"
    try:
        response = requests.get(request_url)
        if response.status_code == 200:
            response_data = response.json()
            ACCESS_TOKEN = response_data.get("access_token")
            TOKEN_RETRIEVED_TIME = time.time()
            return ACCESS_TOKEN
        else:
            print(f"Request failed. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


# Function to split multiple artists
def split_artists(artist_name):
    separators = ["feat.", "featuring", "&", "and", "vs.", "+"]
    for sep in separators:
        if sep in artist_name.lower():
            return [part.strip() for part in re.split(re.escape(sep), artist_name, flags=re.IGNORECASE)]
    return [artist_name]


# Function to preprocess artist names
def generate_search_variations(artist_name):
    variations = set()
    variations.add(artist_name)
    cleaned_name = re.sub(r"[^a-zA-Z0-9\s]", " ", artist_name).strip()
    variations.add(cleaned_name)
    words = cleaned_name.split()
    if len(words) > 1:
        variations.add(words[0])  # First word
        variations.add(words[-1])  # Last word
        variations.add("".join(words))  # Joined without spaces
    return list(variations)


# Function to search for an artist on Spotify
def search_artist(artist_name):
    access_token = get_access_token()
    if not access_token:
        print("Error: No valid access token.")
        return None

    if artist_name in artist_cache:
        return artist_cache[artist_name]

    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {access_token}"}

    artist_variations = generate_search_variations(artist_name)
    for variant in artist_variations:
        params = {"q": f"artist:{variant}", "type": "artist", "limit": 10}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get("artists", {}).get("items"):
                artist_cache[artist_name] = data
                return data

    print(f"No match found for {artist_name}")
    return None


# Function to parse JSON response
def parse_artist_json(json_data):
    if not json_data or not json_data.get('artists') or json_data.get('artists', {}).get('total', 0) == 0:
        return []
    return [
        {"id": artist.get("id"), "name": artist.get("name"), "genres": artist.get("genres", [])}
        for artist in json_data.get('artists', {}).get('items', [])
    ]


# Function to compute similarity
def get_similarity(str1, str2):
    return Levenshtein.ratio(str1.lower(), str2.lower())


# Function to process missing artists
def process_missing_artists():
    try:
        missing_artists_df = pd.read_csv(MISSING_ARTISTS_FILE, sep="\t")
    except Exception as e:
        print(f"Error reading missing artists file: {e}")
        return

    results = []
    for index, row in missing_artists_df.iterrows():
        original_artist_name = row["name"]
        artist_names = split_artists(original_artist_name)

        for artist_name in artist_names:
            search_response = search_artist(artist_name)
            if not search_response:
                continue

            spotify_artists = parse_artist_json(search_response)
            if not spotify_artists:
                continue

            most_similar_artist = max(spotify_artists, key=lambda artist: get_similarity(artist_name, artist["name"]),
                                      default=None)
            if not most_similar_artist:
                continue

            similarity_score = get_similarity(artist_name, most_similar_artist["name"])
            result = {
                "lastfm_artist_id": row["id"],
                "lastfm_artist_name": original_artist_name,
                "spotify_artist_id": most_similar_artist["id"],
                "spotify_artist_name": most_similar_artist["name"],
                "similarity_score": similarity_score
            }
            results.append(result)
            print(result)

    if results:
        results_df = pd.DataFrame(results)
        mode = "a" if os.path.exists(OUTPUT_FILE) else "w"
        header = not os.path.exists(OUTPUT_FILE)
        results_df.to_csv(OUTPUT_FILE, sep="\t", index=False, mode=mode, header=header)
        print(f"Updated results saved to {OUTPUT_FILE}.")
    else:
        print("No new artists processed.")


if __name__ == "__main__":
    process_missing_artists()
