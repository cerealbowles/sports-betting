# Use an official Python runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy Python dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose port 5000
EXPOSE 5000

# Command to run
CMD ["python", "app.py"]

