# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Install pipenv
RUN pip install pipenv

# Copy Pipfile and Pipfile.lock
COPY Pipfile* ./

# Copy the rest of the application
COPY . .

# Install dependencies and run setup
RUN pipenv install --deploy
RUN pipenv run setup

# Expose port 5000
EXPOSE 5000

# Set environment variable to run Flask in production mode
ENV FLASK_ENV=production

# Run the web application
ENTRYPOINT ["pipenv", "run", "start"]
