import pandas as pd
import requests
import Levenshtein

def return_access_token():
    # Define the URL for the request
    request_url = "http://127.0.0.1:5000/get-access-token"

    try:
        # Send a GET request to the specified URL
        response = requests.get(request_url)

        # Check if the request was successful (HTTP status code 200)
        if response.status_code == 200:
            response_data = response.json()  # Assuming the response body is JSON
            access_token = response_data.get("access_token")  # Extract 'access_token'

            if access_token:
                return access_token
            else:
                return response.status_code
        else:
            print(f"Request failed. Status code: {response.status_code}")
            print("Response text:", response.text)
    except requests.exceptions.RequestException as e:
        # Handle any exceptions (e.g., connection errors, timeouts)
        return None

# Function to calculate similarity between two strings
def get_similarity(str1, str2):
    # Calculate the Levenshtein distance similarity score
    return Levenshtein.ratio(str1, str2)

def search_artist(artist_name, access_token):
    # Spotify Search API endpoint
    url = "https://api.spotify.com/v1/search"

    # Define query parameters
    params = {
        "q": f"artist:{artist_name}",  # Query with the artist name
        "type": "artist",  # Search only for artists
        "limit": 1  # Return only the top result
    }

    # Define headers with the access token for authorization
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    # Send GET request
    response = requests.get(url, headers=headers, params=params)

    # Check if the response is successful
    if response.status_code == 200:
        return response.json()  # Return the JSON response
    else:
        # Handle errors
        return {
            "error": f"Failed to fetch artist data. Status Code: {response.status_code}",
            "details": response.json()
        }

def parse_artist_json(json_data):
    # first check if there is a valid data returned from the spotify api
    if json_data.get('artists', {}).get('total', 0) == 0:
        return None

    # Access the first artist in the 'items' list
    first_artist = json_data.get('artists', {}).get('items', [])[0]

    # Extract the 'id', 'name', and 'genres' fields
    artist_id = first_artist.get('id')
    artist_name = first_artist.get('name')
    artist_genres = first_artist.get('genres', [])

    # Return the extracted values
    return {
        "id": artist_id,
        "name": artist_name,
        "genres": artist_genres
    }

def get_random_artist(fpath):
    access_token = return_access_token()
    artist_info = search_artist("Miles Davis", access_token)
    artist_data = pd.read_csv(fpath, sep="\t")
    random_row = artist_data.sample(n=1)
    random_artist_name = random_row["name"].values[0]
    return random_artist_name

def main():
    fpath = "/Users/godfather/Library/CloudStorage/OneDrive-Personal/MacProjects/PycharmProjects/spotifyWebApi/data/lastfmdata/artists.dat"
    access_token = return_access_token()
    random_artist_name = get_random_artist(fpath)
    random_artist_data = parse_artist_json(search_artist(random_artist_name, access_token))
    print(f"Random artist name from the lastfm dataset: {random_artist_name}")
    print(random_artist_data)

if __name__ == "__main__":
    main()