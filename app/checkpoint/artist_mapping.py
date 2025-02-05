import pandas as pd
import requests
import Levenshtein
import time
import os

# Global variables for token and retrieval time
ACCESS_TOKEN = None
TOKEN_RETRIEVED_TIME = None
TOKEN_EXPIRY_TIME = 50 * 60  # 50 minutes in seconds

# Global cache for search results
artist_cache = {}

# File paths
INPUT_FILE = "/data/lastfmdata/artists.dat"
OUTPUT_FILE = "/data/artist_mapping.dat"


# Function to get or refresh the access token
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


# Function to calculate similarity
def get_similarity(str1, str2):
    return Levenshtein.ratio(str1.lower(), str2.lower())


# Function to properly format artist name
def encode_artist_search_query(artist_name):
    cleaned_name = artist_name.replace(".", "").strip()
    return f"artist:{cleaned_name}"


# Function to make a request with rate-limit handling
def make_spotify_request(url, headers, params):
    while True:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 5))
            print(f"Rate limit hit! Waiting {retry_after} seconds...")
            time.sleep(retry_after)
        else:
            return response


# Function to search for an artist on Spotify
def search_artist(artist_name):
    access_token = get_access_token()
    if not access_token:
        print("Error: No valid access token.")
        return None

    if artist_name in artist_cache:
        return artist_cache[artist_name]

    url = "https://api.spotify.com/v1/search"
    params = {"q": encode_artist_search_query(artist_name), "type": "artist", "offset": 0, "limit": 10}
    headers = {"Authorization": f"Bearer {access_token}"}

    response = make_spotify_request(url, headers, params)
    if response.status_code == 200:
        artist_cache[artist_name] = response.json()
        return artist_cache[artist_name]
    else:
        print(f"Error fetching artist: {artist_name}, Status Code: {response.status_code}")
        return None


# Function to parse JSON response
def parse_artist_json(json_data):
    if not json_data or not json_data.get('artists') or json_data.get('artists', {}).get('total', 0) == 0:
        return []

    return [
        {"id": artist.get("id"), "name": artist.get("name"), "genres": artist.get("genres", [])}
        for artist in json_data.get('artists', {}).get('items', [])
    ]


# Function to ensure API requests are spaced out
LAST_REQUEST_TIME = 0
REQUEST_DELAY = 1.5


def throttle_request():
    global LAST_REQUEST_TIME
    elapsed_time = time.time() - LAST_REQUEST_TIME
    if elapsed_time < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - elapsed_time)
    LAST_REQUEST_TIME = time.time()


# Function to load previously processed artists
def load_existing_mappings():
    if os.path.exists(OUTPUT_FILE):
        try:
            return pd.read_csv(OUTPUT_FILE, sep="\t")
        except Exception as e:
            print(f"Error reading output file: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def main():
    print(get_access_token())
    try:
        artist_data = pd.read_csv(INPUT_FILE, sep="\t")
        if "id" not in artist_data.columns or "name" not in artist_data.columns:
            print("Error: Input file must contain 'id' and 'name' columns.")
            return
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    existing_mappings = load_existing_mappings()
    processed_artists = set(existing_mappings["lastfm_artist_id"].astype(str)) if not existing_mappings.empty else set()
    total_artists = len(artist_data)
    half_point = total_artists // 2

    results = []
    for index, row in artist_data.iterrows():
        if str(row["id"]) in processed_artists:
            continue  # Skip already processed artists

        if index >= half_point and not os.path.exists(OUTPUT_FILE):
            break  # Process only half the artists in the first run

        throttle_request()
        spotify_response = search_artist(row["name"])
        spotify_artists = parse_artist_json(spotify_response)

        if not spotify_artists:
            continue

        most_similar_artist = max(spotify_artists, key=lambda artist: get_similarity(row["name"], artist["name"]),
                                  default=None)
        if not most_similar_artist:
            continue

        similarity_score = get_similarity(row["name"], most_similar_artist["name"])
        result = {
            "lastfm_artist_id": row["id"],
            "lastfm_artist_name": row["name"],
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
        print(f"Results successfully saved to {OUTPUT_FILE}.")
    else:
        print("No new artists processed.")


if __name__ == "__main__":
    main()
