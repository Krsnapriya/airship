FROM python:3.11-slim

# Set environment variables for better logging and HF compatibility
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860

# Install system dependencies if needed (e.g. for matplotlib)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN chmod +x start.sh

EXPOSE 7860
EXPOSE 8000

CMD ["./start.sh"]
