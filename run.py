from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 🔥 This enables CORS for all routes
# cors endpoint
@app.route("/recommend", methods=["GET", "OPTIONS"])
def recommend():
    return jsonify({"message": "CORS is now working!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
