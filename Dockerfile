# ---- Base Python image ----
FROM python:3.11-slim

# Work inside /app
WORKDIR /app

# Keep Python fast & clean
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- OS packages (from packages.txt) ----
# If packages.txt doesn't exist or is empty, this still succeeds.
COPY packages.txt ./
RUN apt-get update \
 && if [ -s packages.txt ]; then xargs -a packages.txt apt-get install -y --no-install-recommends; fi \
 && rm -rf /var/lib/apt/lists/*

# ---- Python dependencies ----
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# ---- App code + model weights ----
COPY . .

# Streamlit default port
EXPOSE 8501

# Entry point (change to app_enhanced.py if you prefer)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
