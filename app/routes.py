from flask import Blueprint, jsonify, request
from dotenv import load_dotenv
from app.getToken import get_token
import app.recommender as recommender_module
import logging
import traceback

# Load environment variables from a .env file
load_dotenv()

# Blueprint setup
main = Blueprint('main', __name__)

# Define a scope variable if it will be used in the routes
scope = "user-library-read"

# Configure logging
logging.basicConfig(level=logging.ERROR)


@main.route('/get-access-token', methods=['GET'])
def get_access_token():
    return get_token()


@main.route('/recommend', methods=['GET'])
def recommender_route():
    try:
        # Getting the search parameter
        search_param = request.args.get('search')
        if not search_param:
            return jsonify({'error': 'Missing search parameter'}), 400
        return recommender_module.recommend_based_on_search(search_param)

    except IndexError:
        logging.error("Artist not found in dataset: %s", search_param)
        return jsonify({'error': f'Artist "{search_param}" not found in dataset'}), 404

    except Exception as e:
        logging.error("Unexpected error: %s", traceback.format_exc())
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500


import logging

logging.basicConfig(level=logging.DEBUG)
@main.route('/recommend/genre', methods=['POST'])
def recommend_genre():
    try:
        # Parse incoming JSON request
        data = request.get_json()
        logging.debug(f"Received JSON: {data}")

        if not data or "artists" not in data or "items" not in data["artists"]:
            return jsonify({'error': 'Invalid JSON data, missing "artists.items" key'}), 400

        # Extract artist IDs and genres safely
        user_artist_ids = {
            artist.get("id") for artist in data["artists"]["items"] if artist.get("id")
        }
        user_genres = {
            genre for artist in data["artists"]["items"] if artist.get("genres") for genre in artist["genres"]
        }

        print(f"Extracted user_artist_ids: {user_artist_ids}")
        print(f"Extracted user_genres: {user_genres}")

        # Fetch recommendations for multiple genres
        recommendations = recommender_module.recommend_based_on_genre(list(user_genres), n=10, exclude_ids=user_artist_ids)
        print(f"Recommendations before filtering: {recommendations}")

        filtered_recommendations = []

        for artist_name, spotify_id in recommendations["artist_ids"].items():
            if spotify_id and spotify_id not in user_artist_ids:  # Only add new recommendations
                filtered_recommendations.append({
                    "external_urls": {
                        "spotify": f"https://open.spotify.com/artist/{spotify_id}"
                    },
                    "id": spotify_id,
                    "name": artist_name,
                    "uri": f"spotify:artist:{spotify_id}"
                })

        return jsonify({"recommended_artists": filtered_recommendations})

    except Exception as e:
        logging.error("Unexpected error: %s", traceback.format_exc())
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500


@main.route('/')
def home():
    """
    Simple test route to verify that the API is reachable.
    """
    return "Welcome to the Song Recommendation API", 200
