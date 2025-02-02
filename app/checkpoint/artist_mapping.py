import pandas as pd
import requests
import Levenshtein
import urllib.parse

# Function to get access token from local API
def return_access_token():
    request_url = "http://127.0.0.1:5000/get-access-token"
    try:
        response = requests.get(request_url)
        if response.status_code == 200:
            response_data = response.json()
            return response_data.get("access_token")
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

# Function to search for an artist on Spotify
def search_artist(artist_name, access_token):
    url = "https://api.spotify.com/v1/search"

    # Define query parameters (NO manual encoding here)
    params = {
        "q": encode_artist_search_query(artist_name),  # Pass raw formatted query
        "type": "artist",
        "offset": 0,  # Start from the most relevant results
        "limit": 10
    }

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        print(f"Sending a request to this URL: {response.url}")  # Debugging output

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: Failed to fetch artist data. Status Code: {response.status_code}")
            print("Response text:", response.text)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
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

def main():
    # Path to the Last.fm artists file
    fpath = "/Users/godfather/Library/CloudStorage/OneDrive-Personal/MacProjects/PycharmProjects/spotifyWebApi/data/lastfmdata/artists.dat"  # Update as needed to match the file location
    output_file = "artist_mapping.dat"  # Output file to save the results

    # Load artists from .dat file
    try:
        artist_data = pd.read_csv(fpath, sep="\t")  # Assume tab-separated file
        if "id" not in artist_data.columns or "name" not in artist_data.columns:
            print("Error: The artists.dat file must have 'id' and 'name' columns.")
            return
    except Exception as e:
        print(f"Error reading artists file: {e}")
        return

    # Get access token
    access_token = return_access_token()
    if not access_token:
        print("Error: No access token received.")
        return

    # Store results
    results = []

    # Iterate through each artist in the file
    for index, row in artist_data.iterrows():
        lastfm_artist_id = row["id"]
        lastfm_artist_name = row["name"]

        # Search for the artist on Spotify
        spotify_response = search_artist(lastfm_artist_name, access_token)
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
