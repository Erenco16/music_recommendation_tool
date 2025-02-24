from flask import Blueprint, request, jsonify
from dotenv import load_dotenv
from app.getToken import get_token
import app.artist_mapping.recommender as recommender_module
import os

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
    artists, scores = recommender_module.main()
    return jsonify({'artists': artists, 'scores': scores.tolist()})


@main.route('/')
def home():
    """
    Simple test route to verify that the API is reachable.
    """
    return "Welcome to the Song Recommendation API", 200
