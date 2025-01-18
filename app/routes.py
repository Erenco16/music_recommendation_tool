from flask import Blueprint, request, jsonify
from dotenv import load_dotenv
from app.getToken import get_token
import os

# Load environment variables from a .env file
load_dotenv()

# Blueprint setup
main = Blueprint('main', __name__)

# Define a scope variable if it will be used in the routes
scope = "user-library-read"


@main.route('/song-details', methods=['GET'])
def get_song_details():
    return "That's how you can get the song details", 200

@main.route('/get-access-token', methods = ['GET'])
def get_access_token():
    return get_token()

@main.route('/')
def home():
    """
    Simple test route to verify that the API is reachable.
    """
    return "Welcome to the Song Recommendation API", 200
