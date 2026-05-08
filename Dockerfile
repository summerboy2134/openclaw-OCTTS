FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list 2>/dev/null || true \
  && apt-get update \
  && apt-get install -y --no-install-recommends build-essential gfortran libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./

# Install third-party dependencies before copying application code.
# This layer is reused when only src/ or ops/ changes, avoiding repeated downloads
# of large wheels such as xgboost's Linux dependencies.
RUN mkdir -p src/octts \
  && touch src/octts/__init__.py \
  && pip install --no-cache-dir --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ \
  && pip install --no-cache-dir . -i https://mirrors.aliyun.com/pypi/simple/ \
  && rm -rf src

COPY src ./src
COPY ops ./ops

RUN pip install --no-cache-dir --no-deps . -i https://mirrors.aliyun.com/pypi/simple/

RUN mkdir -p /app/memory

EXPOSE 8000

CMD ["uvicorn", "octts.api:app", "--host", "0.0.0.0", "--port", "8000"]
