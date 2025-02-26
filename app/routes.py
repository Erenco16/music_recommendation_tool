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

        artists, scores = recommender_module.recommend_based_on_artist(search_param)
        return jsonify({'artists': artists, 'scores': scores.tolist()})

    except IndexError:
        logging.error("Artist not found in dataset: %s", search_param)
        return jsonify({'error': f'Artist "{search_param}" not found in dataset'}), 404

    except Exception as e:
        logging.error("Unexpected error: %s", traceback.format_exc())
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500


@main.route('/')
def home():
    """
    Simple test route to verify that the API is reachable.
    """
    return "Welcome to the Song Recommendation API", 200
