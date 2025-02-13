from pathlib import Path

# Diretório base do projeto
BASE_DIR = Path(__file__).parent

# Mapeamento das classes detectadas para nomes de poses
CLASSES_DETECTADAS = {
    0: "em pé",
    1: "sentado",
    2: "deitado"
}

# Configurações padrão
DURACAO_PADRAO = 300  # 5 minutos em segundos
ARQUIVO_PESOS = str(BASE_DIR / "best.pt")
FPS_PADRAO = 30
ARQUIVO_VIDEO_PADRAO = str(BASE_DIR / "video_idoso.mp4")

# Estados de pose inicial
POSE_NAO_DETECTADA = "não detectado"

# Configurações de visualização
COR_BBOX = (0, 255, 0)  # Verde em BGR
ESPESSURA_BBOX = 2
TAMANHO_FONTE = 0.5
