# 1. Use a slim base image to keep the size small
FROM python:3.11-slim

# 2. Install curl for the healthcheck, then clean up to save space
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory
WORKDIR /app

# 4. Copy ONLY requirements first (Layer Cache optimization)
COPY requirements.txt .

# 5. Install dependencies. --no-cache-dir keeps the image small
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the application code
COPY . .

# 7. Create a non-root user and give them ownership of the app directory
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# 8. Expose the port the app runs on
EXPOSE 5000

# 9. Add a healthcheck to ensure the Flask app is responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# 10. Command to run the application
CMD ["python", "app.py"]