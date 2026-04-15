
# Dockerfile


FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for PyMuPDF and OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caches this layer separately)
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir keeps image size smaller
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model during build
RUN python -m spacy download en_core_web_sm

# Copy project files
COPY . .

# Create directories for runtime data
RUN mkdir -p data/chroma_db data/uploads

# Expose Streamlit's default port
EXPOSE 8501

# Streamlit configuration for containerized environments
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Start the app
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0"]
