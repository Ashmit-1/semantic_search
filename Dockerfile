FROM python:3.12-slim

# Set working directory to the root of your project
WORKDIR /app

# Install system dependencies for FAISS
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set PYTHONPATH so 'api/main.py' can find 'semantic_cache.py' in the root
ENV PYTHONPATH=/app

# Expose FastAPI port
EXPOSE 8000

# Start command from the root
# Note: we use api.main:app because main.py is inside the api folder
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]