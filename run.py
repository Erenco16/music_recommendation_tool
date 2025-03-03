from flask import Flask
from flask_cors import CORS
from app.routes import main  # Import the blueprint

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers=["Content-Type", "Authorization"], supports_credentials=True)

# Register the blueprint from routes.py
app.register_blueprint(main)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)