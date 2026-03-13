FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY ops ./ops

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir .

RUN mkdir -p /app/memory

EXPOSE 8000

CMD ["uvicorn", "octts.api:app", "--host", "0.0.0.0", "--port", "8000"]
