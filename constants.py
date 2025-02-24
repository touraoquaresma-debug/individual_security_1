from pathlib import Path

# Diretório base do projeto
BASE_DIR = Path(__file__).parent

# Mapeamento das classes detectadas para nomes de poses
CLASSES_DETECTADAS = {
    0: 'idoso deitado',
    1: 'idoso em pe',
    2: 'idoso sentado',
    3: 'jovem'
}

# Configurações padrão
DURACAO_PADRAO = 300  # 5 minutos em segundos
ARQUIVO_PESOS = str(BASE_DIR / r'runs\pose\train\weights\best.pt')
ARQUIVO_VIDEO_PADRAO = str(BASE_DIR / 'video_idoso.mp4')
ARQUIVO_CONFIGURACAO_DATASET = str(
    BASE_DIR / 'downloads/YOLOElderlyPose.v2i.yolov11/data.yaml'
)

# Estados de pose inicial
POSE_NAO_DETECTADA = 'não detectado'

# Configurações de visualização
COR_BBOX = (0, 255, 0)  # Verde em BGR
ESPESSURA_BBOX = 2
TAMANHO_FONTE = 0.5
