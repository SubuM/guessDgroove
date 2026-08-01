FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY v13.py .

RUN mkdir -p cache/v7

EXPOSE 8501

CMD ["streamlit", "run", "v13.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
