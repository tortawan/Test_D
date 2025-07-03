# Use Python 3.11 slim as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Install system dependencies for Box2D and graphics
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python script
COPY lunar_lander_dqn.py .

# Create directories for outputs
RUN mkdir -p /app/outputs

# Set the default command
CMD ["python", "lunar_lander_dqn.py"]
