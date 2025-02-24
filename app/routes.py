from flask import Blueprint, jsonify, request
from dotenv import load_dotenv
from app.getToken import get_token
import app.recommender as recommender_module

# Load environment variables from a .env file
load_dotenv()

# Blueprint setup
main = Blueprint('main', __name__)

# Define a scope variable if it will be used in the routes
scope = "user-library-read"


@main.route('/get-access-token', methods = ['GET'])
def get_access_token():
    return get_token()

@main.route('/recommend', methods = ['GET'])
def recommender_route():
    # getting the search parameter
    search_param = request.args.get('search')
    artists, scores = recommender_module.recommend_based_on_artist(search_param)
    return jsonify({'artists': artists, 'scores': scores.tolist()})


@main.route('/')
def home():
    """
    Simple test route to verify that the API is reachable.
    """
    return "Welcome to the Song Recommendation API", 200
