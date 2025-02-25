# Sistema de Monitoramento de Poses para Idosos 🤖👴

Sistema de monitoramento de poses baseado em YOLO para detecção de posturas de idosos, capaz de identificar se a pessoa está idoso em pe, sentada ou deitada.

## ✨ Funcionalidades

- 📹 Suporte para câmera web ou arquivo de vídeo
- ⏱️ Monitoramento por tempo determinado
- 🎯 Detecção de 3 poses: idoso em pe, idoso sentado e idoso deitado
- 📊 Relatório detalhado com duração de cada pose
- 🔄 Interface interativa via terminal
- 🐳 Suporte a Docker
- 💻 Executável standalone disponível

## 🚀 Como Usar

### Opção 1: Usando Python (Recomendado para Desenvolvedores)

#### Pré-requisitos
- Python 3.12 ou superior
- Git
- uv (recomendado para gerenciamento de dependências)

#### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/joaosnet/YOLO-Elderly-Pose-Detection-Monitoring.git
cd YOLO-Elderly-Pose-Detection-Monitoring
```

2. Instale o uv (opcional, mas recomendado):
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Instale as dependências:
```bash
# Usando uv (recomendado)
uv sync

# Ou usando pip tradicional
pip install -m requirements.txt
```

4. Para desenvolvimento, instale as dependências de desenvolvimento:
```bash
# Usando uv
uv sync --dev
```

5. Ative o ambiente virtual:
```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/Scripts/activate
```


#### Comandos Disponíveis

O projeto utiliza `taskipy` para automatizar comandos comuns. Aqui estão os comandos disponíveis:

```bash
# Executar o monitoramento (modo interativo)
task run

# Formatar e verificar código
task format
task lint

# Iniciar treinamento
task train

# Iniciar Jupyter Lab
task jupyter
```

### Opção 2: Usando Docker 🐳

1. Construa a imagem:
```bash
docker build -t elderly-monitoring .
```

2. Execute o container:
```bash
docker run elderly-monitoring
```

Para usar um vídeo específico, monte um volume:
```bash
docker run -v $(pwd):/app elderly-monitoring --video_path seu_video.mp4
```

### Opção 3: Executável Standalone 📦

1. Baixe o último release na seção "Releases"
2. Extraia o arquivo zip (se estiver comprimido)
3. Execute o arquivo `elderly-pose-windows-2019` (Windows), `elderly-pose-ubuntu-22.04` (Linux) ou `elderly-pose-macos-latest` (Mac)

## 📊 Estrutura do Projeto

```
YOLO-Elderly-Pose-Detection-Monitoring/
├── main.py           # Script principal
├── constants.py      # Constantes e configurações
├── best.pt          # Arquivo de pesos do modelo
├── pyproject.toml   # Configurações do projeto
├── Dockerfile       # Configuração Docker
└── README.md        # Este arquivo
```

## 🛠️ Configurações

O sistema possui várias configurações que podem ser ajustadas:

- `DURACAO_PADRAO`: 300 segundos (5 minutos)
- Classes detectadas:
  - 0: idoso em pe
  - 1: idoso sentado
  - 2: idoso deitado
  - 3: jovem

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## 👥 Contribuindo

Contribuições são bem-vindas! Por favor, leia o arquivo CONTRIBUTING.md antes de enviar um Pull Request.

## 🐛 Problemas Conhecidos

Se encontrar algum problema, por favor [abra uma issue](https://github.com/seu-usuario/YOLO-Elderly-Pose-Detection-Monitoring/issues).

## 📞 Contato

Para dúvidas ou sugestões, entre em contato através das issues do GitHub.
