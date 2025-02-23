import base64
import requests
import os
from dotenv import load_dotenv
from app.db_operations import readData
import json
import pandas as pd

load_dotenv()

# Spotify client credentials
client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')


def get_token():
    # Encode the client credentials as a base64 string
    auth_str = f"{client_id}:{client_secret}"
    auth_bytes = auth_str.encode('utf-8')
    auth_base64 = base64.b64encode(auth_bytes).decode('utf-8')

    # Define the request parameters
    auth_url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Authorization': f'Basic {auth_base64}'
    }
    data = {
        'grant_type': 'client_credentials'
    }

    # Make the POST request to get the access token
    response = requests.post(auth_url, headers=headers, data=data)


    return response.text



def get_auth_header():
    return {"Authorization": "Bearer " + get_token()}


def search_for_artist(artist_name):
    search_url = "https://api.spotify.com/v1/search?"
    headers = get_auth_header()
    search_query = f"q={artist_name}&type=artist&limit=1"
    result = requests.get(search_url+search_query, headers=headers)
    json_result = json.loads(result.content)
    if len(json_result) == 0:
        return f"No artists found with the name: {artist_name}"
    else:
        return json_result["artists"]["items"]

def main():
    test_df = readData.read_db(1)
    artist_name = test_df.artist_name
    artist_info = search_for_artist(artist_name)[0]
    artist_url = artist_info["external_urls"]["spotify"]
    print(artist_info)


if __name__ == '__main__':
    main()