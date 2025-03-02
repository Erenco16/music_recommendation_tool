from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)  # Allow all origins

@app.route("/recommend", methods=["GET", "OPTIONS"])
def recommend():
    return jsonify({"message": "CORS is working!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
