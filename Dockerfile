# Dockerfile

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Create a non-root user (UID 1000) for security and Hugging Face Spaces compatibility
RUN useradd -m -u 1000 appuser

# Install system dependencies needed for PyMuPDF and OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
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

# Copy project files
COPY . .

# Create directories for runtime data and set ownership to the non-root user
RUN mkdir -p data/chroma_db data/uploads && \
    chown -R appuser:appuser /app/data

# Switch to the non-root user
USER appuser

ENV PYTHONUNBUFFERED=1

# Streamlit configuration for containerized environments
# Make the port configurable (defaults to 8501, Hugging Face Spaces uses 7860)
ENV PORT=8501
ENV STREAMLIT_SERVER_PORT=${PORT}
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Start the app
CMD streamlit run ui/app.py --server.address=0.0.0.0 --server.port=${PORT}
