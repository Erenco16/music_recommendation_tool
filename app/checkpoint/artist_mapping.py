import pandas as pd
import requests
import Levenshtein
import urllib.parse
import time

# Global variables for token and retrieval time
ACCESS_TOKEN = None
TOKEN_RETRIEVED_TIME = None
TOKEN_EXPIRY_TIME = 50 * 60  # 50 minutes in seconds

# Global cache for search results
artist_cache = {}

# Function to get or refresh the access token
def get_access_token():
    global ACCESS_TOKEN, TOKEN_RETRIEVED_TIME

    # Check if the token is still valid
    if ACCESS_TOKEN and TOKEN_RETRIEVED_TIME and (time.time() - TOKEN_RETRIEVED_TIME < TOKEN_EXPIRY_TIME):
        return ACCESS_TOKEN  # Return cached token if still valid

    # Otherwise, request a new access token
    request_url = "http://127.0.0.1:5000/get-access-token"
    try:
        response = requests.get(request_url)
        if response.status_code == 200:
            response_data = response.json()
            ACCESS_TOKEN = response_data.get("access_token")
            TOKEN_RETRIEVED_TIME = time.time()  # Update retrieval time
            return ACCESS_TOKEN
        else:
            print(f"Request failed. Status code: {response.status_code}")
            print("Response text:", response.text)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Function to calculate similarity between two strings
def get_similarity(str1, str2):
    return Levenshtein.ratio(str1.lower(), str2.lower())  # Case insensitive comparison

# Function to properly format the artist name for Spotify search
def encode_artist_search_query(artist_name):
    cleaned_name = artist_name.replace(".", "").strip()  # Remove dots and strip spaces
    return f"artist:{cleaned_name}"  # No encoding needed!

# Function to make a request and handle rate limits
def make_spotify_request(url, headers, params):
    while True:
        response = requests.get(url, headers=headers, params=params)

        # Handle rate limits (429 Too Many Requests)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 5))  # Default to 5 sec if missing
            print(f"Rate limit hit! Waiting {retry_after} seconds before retrying...")
            time.sleep(retry_after)
        else:
            return response

# Function to search for an artist on Spotify with caching and rate-limit handling
def search_artist(artist_name):
    access_token = get_access_token()
    if not access_token:
        print("Error: No valid access token.")
        return None

    # Check if artist is in cache
    if artist_name in artist_cache:
        print(f"Using cached result for {artist_name}")
        return artist_cache[artist_name]

    url = "https://api.spotify.com/v1/search"
    params = {
        "q": encode_artist_search_query(artist_name),
        "type": "artist",
        "offset": 0,
        "limit": 10
    }

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    # Make API request with rate-limit handling
    response = make_spotify_request(url, headers, params)

    if response.status_code == 200:
        artist_cache[artist_name] = response.json()  # Save result in cache
        return artist_cache[artist_name]
    else:
        print(f"Error: Failed to fetch artist data. Status Code: {response.status_code}")
        print("Response text:", response.text)
        return None

# Function to parse the JSON response from Spotify
def parse_artist_json(json_data):
    if not json_data or not json_data.get('artists') or json_data.get('artists', {}).get('total', 0) == 0:
        return []

    artists = json_data.get('artists', {}).get('items', [])
    return [
        {
            "id": artist.get("id"),
            "name": artist.get("name"),
            "genres": artist.get("genres", [])
        }
        for artist in artists
    ]

# Function to get a random artist from Last.fm dataset
def get_random_artist(fpath):
    artist_data = pd.read_csv(fpath, sep="\t")
    random_row = artist_data.sample(n=1)
    return random_row["name"].values[0]

# Function to ensure API requests are spaced out
LAST_REQUEST_TIME = 0
REQUEST_DELAY = 1.5  # Wait time between requests (1.5 sec)

def throttle_request():
    global LAST_REQUEST_TIME
    elapsed_time = time.time() - LAST_REQUEST_TIME
    if elapsed_time < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - elapsed_time)  # Wait to prevent rate limits
    LAST_REQUEST_TIME = time.time()  # Update last request time

def main():
    fpath = "/Users/godfather/Library/CloudStorage/OneDrive-Personal/MacProjects/PycharmProjects/spotifyWebApi/data/lastfmdata/artists.dat"
    output_file = "artist_mapping.dat"

    # Load artists from .dat file
    try:
        artist_data = pd.read_csv(fpath, sep="\t")
        if "id" not in artist_data.columns or "name" not in artist_data.columns:
            print("Error: The artists.dat file must have 'id' and 'name' columns.")
            return
    except Exception as e:
        print(f"Error reading artists file: {e}")
        return

    # Store results
    results = []

    # Iterate through each artist in the file
    for index, row in artist_data.iterrows():
        lastfm_artist_id = row["id"]
        lastfm_artist_name = row["name"]

        # Ensure we don't hit the API too fast
        throttle_request()

        # Search for the artist on Spotify
        spotify_response = search_artist(lastfm_artist_name)
        spotify_artists = parse_artist_json(spotify_response)

        # If no artists are found, skip
        if not spotify_artists:
            print(f"No related artist found for Last.fm artist: {lastfm_artist_name}")
            continue

        # Find the most similar artist
        most_similar_artist = max(
            spotify_artists,
            key=lambda artist: get_similarity(lastfm_artist_name, artist["name"]),
            default=None
        )

        # If no similar artist is found, skip
        if not most_similar_artist:
            print(f"No similar artist found on Spotify for Last.fm artist: {lastfm_artist_name}")
            continue

        # Calculate similarity score
        similarity_score = get_similarity(lastfm_artist_name, most_similar_artist["name"])

        # Collect relevant data
        result = {
            "lastfm_artist_id": lastfm_artist_id,
            "lastfm_artist_name": lastfm_artist_name,
            "spotify_artist_id": most_similar_artist["id"],
            "spotify_artist_name": most_similar_artist["name"],
            "similarity_score": similarity_score
        }
        results.append(result)

        # Print progress
        print(f"Processed: Last.fm '{lastfm_artist_name}' -> Spotify '{most_similar_artist['name']}'")

    # Save results to a .dat file
    results_df = pd.DataFrame(results)
    try:
        results_df.to_csv(output_file, sep="\t", index=False)
        print(f"Results successfully saved to {output_file}.")
    except Exception as e:
        print(f"Error saving results to file: {e}")

if __name__ == "__main__":
    main()
