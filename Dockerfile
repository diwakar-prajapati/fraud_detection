FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements_render.txt .
RUN pip install --no-cache-dir -r requirements_render.txt

# Copy all application code
COPY application/ ./application/
COPY models/ ./models/
COPY src/ ./src/
COPY main.py .
COPY run_all.py .

# Create necessary directories
RUN mkdir -p models plots

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "application.bank_api:app", "--host", "0.0.0.0", "--port", "8000"]