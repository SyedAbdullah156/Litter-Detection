FROM python:3.10-slim

WORKDIR /app

# Copy files
COPY index.html .
COPY style.css .

# Expose port
EXPOSE 3000

# Simple HTTP server
CMD ["python", "-m", "http.server", "3000"]