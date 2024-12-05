FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
  && apt-get install --yes --no-install-recommends \
    curl \
  && apt-get autoremove \
  && rm -rf /var/lib/apt/lists/*

ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VERSION=1.8.4
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_CACHE_DIR=/tmp/poetry_cache
ENV PATH="/opt/poetry/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /code
COPY poetry.lock pyproject.toml /code/
COPY pyproject.toml /code/
RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR
COPY . /code/
