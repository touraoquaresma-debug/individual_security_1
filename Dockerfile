FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Instala dependências do sistema necessárias para o OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app
RUN uv sync --all-extras --dev --frozen

ENV PATH="/app/.venv/bin:$PATH"

# Presuming there is a `my_app` command provided by the project
CMD ["task", "run"]
