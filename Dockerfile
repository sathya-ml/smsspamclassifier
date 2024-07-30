# Stage 1: Build stage
FROM python:3.11-slim AS build

# Prevent Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure stdout and stderr are flushed immediately
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./src .
COPY ./model/model.pkl ./model.pkl

# Expose the port
EXPOSE 8080

# Run the Flask application - development
# CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]

# Run the Flask application with Gunicorn - production
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]