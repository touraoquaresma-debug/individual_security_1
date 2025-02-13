FROM ghcr.io/astral-sh/uv:python3.12-slim-bookworm

# Instala dependências do sistema necessárias para o OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto
COPY pyproject.toml main.py constants.py ./
COPY best.pt ./

# Configura o ambiente UV e instala as dependências
ENV UV_SYSTEM_PYTHON=1
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# Comando para executar o programa
CMD ["python", "main.py"]
