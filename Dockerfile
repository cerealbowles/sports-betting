FROM python:3.11-slim

WORKDIR /app

# Dependencies first — better layer caching on rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Ensure data directory exists at image build time
RUN mkdir -p /data

EXPOSE 5000

# Single worker so the in-memory cache (warmed by APScheduler) is shared
# across all requests. A personal home server never needs more than 1 worker.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
