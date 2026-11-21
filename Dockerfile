# Use a lightweight Python image
FROM python:slim

# Prevent .pyc files and keep Python output unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only requirements first (better layer caching)
COPY requirements.txt requirements.txt

# Install packages without caching to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Train the model at build time so the model file is baked into the image
RUN python train.py

EXPOSE 5000

CMD ["python", "app.py"]
