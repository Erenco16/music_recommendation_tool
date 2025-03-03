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
        result = recommender_module.recommend_based_on_search(search_param)
        if not result:
            return jsonify({'error': 'No artists found with the given search parameter'}), 400

    except IndexError:
        logging.error("Artist not found in dataset: %s", search_param)
        return jsonify({'error': f'Artist "{search_param}" not found in dataset'}), 404

    except Exception as e:
        logging.error("Unexpected error: %s", traceback.format_exc())
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500


def clean_recommendations(recommended_artists):
    """
    Cleans the recommended artist list by:
    - Removing artists with 'Unknown ID'
    - Fixing 'Unknown Artist' names if a valid ID is present
    """
    cleaned_artists = []

    for artist in recommended_artists:
        artist_ids = artist.get("id", [])
        # Skip artists if "Unknown ID" is in the IDs
        if "Unknown ID" in artist_ids:
            continue
        # If the name is "Unknown Artist" but a valid ID exists, use the valid ID as the name
        if artist["name"] == "Unknown Artist" and len(artist_ids) > 0:
            artist["name"] = artist_ids[0]
        cleaned_artists.append(artist)

    return cleaned_artists



@main.route('/recommend/genre', methods=['POST'])
def recommend_genre():
    try:
        data = request.get_json()
        user_genres = set()
        user_artist_ids = set()

        # Extract user genres and artist IDs from Spotify data
        for artist in data.get("artists", {}).get("items", []):
            user_artist_ids.add(artist["id"])  # Followed artist IDs from Spotify
            user_genres.update(artist.get("genres", []))  # Genres from user's followed artists

        # Get recommendations (this returns a dictionary with keys:
        # "matched_genres", "recommended_artists" (a set of artist names),
        # and "artist_ids" (a dict mapping artist names to Spotify IDs))
        recommendations = recommender_module.recommend_based_on_genre(list(user_genres), n=10, exclude_ids=user_artist_ids)
        artist_mapping = recommender_module.load_artist_metadata()

        # Build a list of raw recommendations from the recommended artist names
        raw_recommendations = []
        for artist_name in recommendations["recommended_artists"]:
            # Use the Spotify ID from the recommendation dict (if found); otherwise "Unknown ID"
            spotify_id = recommendations["artist_ids"].get(artist_name, "Unknown ID")
            # Here we form the structure: "id": [artist_name, spotify_id] (as in your previous structure)
            raw_recommendations.append({"name": artist_name, "id": [artist_name, spotify_id]})

        # Clean the recommendations: remove entries with "Unknown ID" and fix names if possible
        cleaned_recommendations = clean_recommendations(raw_recommendations)

        return jsonify({
            "matched_genres": recommendations["matched_genres"],
            "recommended_artists": cleaned_recommendations
        })

    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error", "message": str(e)}), 500




@main.route('/')
def home():
    """
    Simple test route to verify that the API is reachable.
    """
    return "Welcome to the Song Recommendation API", 200
