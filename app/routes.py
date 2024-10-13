from flask import Blueprint, request, jsonify
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

scope = "user-library-read"
main = Blueprint('main', __name__)

# Spotify authentication setup
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))


@main.route('/song-details', methods=['POST'])
def song_details():
    data = request.json
    track_id = data.get('track_id')
    if not track_id:
        return jsonify({'error': 'track_id is required'}), 400

    track = sp.track(track_id)
    return jsonify(track)
