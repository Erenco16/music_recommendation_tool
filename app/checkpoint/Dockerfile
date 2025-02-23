# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy the necessary files
COPY requirements.txt /app/
COPY app/checkpoint/artist_mapping.py /app/
COPY app/checkpoint/missing_artists.dat /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Wait for the Flask server to be available
CMD ["sh", "-c", "until curl -s http://flask-server:5001/health; do sleep 5; done && python artist_mapping.py"]
