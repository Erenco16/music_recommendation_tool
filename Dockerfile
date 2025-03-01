# Base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the project into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask's port
EXPOSE 5001

# Health check for service readiness
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5001/health || exit 1

# Run Flask server
CMD ["python", "run.py"]
