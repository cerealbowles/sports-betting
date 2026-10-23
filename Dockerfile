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

# 3 workers: a single sync worker meant one slow/blocked request (or the
# startup/background calibration work sharing its process+GIL) stalled every
# other request behind it. The in-memory calibration caches this used to
# justify pinning to 1 worker for are no longer the blocker that pattern was
# written for — see app.py's _load_all_calibration_snapshots()/
# _is_scheduler_leader(): every worker loads its own copy from the shared
# snapshot files under /data at boot (cheap, correct), and only one worker
# (elected via a flock, not necessarily worker 1) runs the APScheduler
# cron + background warm-cycle thread, so odds fetches/ESPN calls/DB writes
# still happen once per container, not once per worker. --worker-class
# gthread makes --threads actually take effect (gunicorn's default sync
# worker class ignores it).
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "3", \
     "--worker-class", "gthread", \
     "--threads", "4", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
