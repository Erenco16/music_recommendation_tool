from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)

    # Enable CORS (Cross-Origin Resource Sharing)
    CORS(app)

    # Register Blueprint from routes.py
    from app.routes import main  # ✅ Import Blueprint from routes.py
    app.register_blueprint(main)  # ✅ Register Blueprint

    @app.route('/health')
    def health_check():
        return "OK", 200

    return app
