FROM python:3.11-slim

WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY apps/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY . /workspace
WORKDIR /workspace/apps/api
ENV PYTHONPATH=/workspace
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
