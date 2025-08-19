import os
import sys
import time
import csv
import json
import requests

import cv2
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.prompt import Confirm, Prompt
from rich.table import Table
from ultralytics import YOLO

from constants import (
    ARQUIVO_PESOS,
    ARQUIVO_VIDEO_PADRAO,
    CLASSES_DETECTADAS,
    DURACAO_PADRAO,
    POSE_NAO_DETECTADA,
)

# Caminho para o arquivo de configurações
CONFIG_FILE = "config.json"

# Tentativa de importar o módulo msvcrt
try:
    import msvcrt
except Exception:
    msvcrt = None

# Verifica se está executando como frozen app
if getattr(sys, 'frozen', False):
    try:
        import pyi_splash  # type: ignore
    except ModuleNotFoundError:
        pass


def initialize_app():
    if 'pyi_splash' in sys.modules:
        pyi_splash.update_text('Carregando módulos...')
        time.sleep(1)
        pyi_splash.update_text('Inicializando interface...')
        time.sleep(1)


console = Console()


def load_config():
    """
    Carrega as configurações do arquivo JSON se ele existir.
    Caso contrário, retorna um dicionário vazio.
    """
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_config(config):
    """
    Salva as configurações no arquivo JSON.
    """
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def get_user_parameters(config):
    """
    Obtém as configurações do usuário. Se não estiverem no arquivo de configuração,
    solicita ao usuário e salva para uso futuro.
    """
    # Se as configurações já estiverem salvas, usaremos elas
    midia_path = config.get('midia_path', '0')
    weights_path = config.get('weights_path', ARQUIVO_PESOS)
    annotated_frame_cv2 = config.get('annotated_frame_cv2', True)
    output_dir = config.get('output_dir', os.path.join('.', 'relatorios'))
    server_url = config.get('server_url', 'http://localhost:8000/upload')

    # Perguntas ao usuário caso não existam configurações
    if midia_path == '0':
        video_source = Prompt.ask(
            '\n📹 Escolha a fonte',
            choices=['webcam', 'video', 'imagem'],
            default='webcam',
        )
        if video_source == 'webcam':
            midia_path = '0'
        elif video_source == 'video':
            midia_path = Prompt.ask(
                '🎥 Digite o caminho do arquivo de vídeo',
                default=ARQUIVO_VIDEO_PADRAO,
            )
        else:  # imagem
            midia_path = Prompt.ask('🖼️ Digite o caminho da imagem (jpg, png, etc.)')

    if weights_path == ARQUIVO_PESOS:
        use_default_weights = Confirm.ask('\n🎯 Usar arquivo de pesos padrão?', default=True)
        if not use_default_weights:
            weights_path = Prompt.ask('Digite o caminho para o arquivo de pesos personalizado')

    annotated_frame_cv2 = Confirm.ask(
        '\n🖼️ Visualizar frames com anotações?', default=True
    )

    if not output_dir:
        output_dir = Prompt.ask(
            '\n📁 Diretório para salvar o CSV/JSON',
            default=os.path.join('.', 'relatorios'),
        )

    if not server_url:
        server_url = Prompt.ask(
            '\n🌐 URL do servidor para enviar o JSON',
            default='http://localhost:8000/upload',
        )

    # Salvar as configurações para as próximas execuções
    config['midia_path'] = midia_path
    config['weights_path'] = weights_path
    config['annotated_frame_cv2'] = annotated_frame_cv2
    config['output_dir'] = output_dir
    config['server_url'] = server_url

    # Salva as configurações
    save_config(config)

    return midia_path, weights_path, annotated_frame_cv2, output_dir, server_url


def eh_imagem(caminho: str) -> bool:
    ext = os.path.splitext(caminho)[1].lower()
    return ext in ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


def salvar_csv_json_relatorio(pose_durations, total_time, csv_dir, origem='midia'):
    """
    Salva a tabela final como CSV e JSON no diretório informado.
    Colunas: pose, duracao_min, porcentagem
    """
    os.makedirs(csv_dir, exist_ok=True)
    total_frames = sum(pose_durations.values())

    # evita divisão por zero
    if total_frames == 0:
        total_frames = 1

    linhas = []
    for pose, frames in pose_durations.items():
        duracao_min = (frames / total_frames) * (total_time / 60) if total_time > 0 else 0.0
        porcentagem = (frames / total_frames) * 100
        linhas.append({'pose': pose, 'duracao_min': f'{duracao_min:.2f}', 'porcentagem': f'{porcentagem:.1f}%'})

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    nome_arquivo_csv = f'relatorio_poses_{origem}_{timestamp}.csv'
    nome_arquivo_json = f'relatorio_poses_{origem}_{timestamp}.json'
    
    caminho_csv = os.path.join(csv_dir, nome_arquivo_csv)
    caminho_json = os.path.join(csv_dir, nome_arquivo_json)

    # Salvar como CSV
    with open(caminho_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['pose', 'duracao_min', 'porcentagem'], delimiter=';')
        writer.writeheader()
        writer.writerows(linhas)

    # Salvar como JSON
    with open(caminho_json, mode='w', encoding='utf-8') as f:
        json.dump(linhas, f, indent=4, ensure_ascii=False)

    return caminho_csv, caminho_json


def enviar_json_para_servidor(json_path, server_url):
    """
    Envia o arquivo JSON para o servidor via HTTP POST.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    try:
        response = requests.post(server_url, json=data)
        if response.status_code == 200:
            console.print(f'✅ JSON enviado com sucesso para {server_url}')
        else:
            console.print(f'[bold red]❌ Erro ao enviar JSON para o servidor! Status: {response.status_code}[/]')
    except Exception as e:
        console.print(f'[bold red]❌ Erro ao tentar enviar o JSON para o servidor: {str(e)}[/]')


def run_pose_monitoring(  # noqa: PLR0912, PLR0914, PLR0915
    midia_path=ARQUIVO_VIDEO_PADRAO,
    weights_path=ARQUIVO_PESOS,
    annotated_frame_cv2=True,
    output_dir='./relatorios',
    server_url='http://localhost:8000/upload',
):
    # Inicialização do monitoramento
    console.print(
        Panel('🎥 Sistema de Monitoramento de Poses', style='bold blue')
    )

    start_time = time.time()
    pose_durations = {
        'idoso deitado': 0,
        'idoso em pe': 0,
        'idoso sentado': 0,
        'jovem': 0,
        POSE_NAO_DETECTADA: 0,
    }
    frame_count = 0

    # Carregamento do modelo YOLO
    with console.status('[bold green]Carregando o modelo YOLO...'):
        try:
            model = YOLO(weights_path)
            model.classes = [0, 1, 2, 3]  # Limite às classes de interesse
            model.conf = 0.5
        except Exception as e:
            console.print(f'[bold red]❌ Erro ao carregar o modelo:[/] {str(e)}')
            return

    # Modo WEBCAM/VÍDEO
    cap = cv2.VideoCapture(0 if midia_path == '0' else midia_path)
    if not cap.isOpened():
        console.print('[bold red]❌ Erro ao abrir fonte de vídeo!')
        return

    console.print('\n📊 Configurações:')
    console.print(f'- Captura durante 20 segundos com intervalo de 1 segundo entre cada frame')
    console.print(f'- Classes detectadas: {[CLASSES_DETECTADAS[c] for c in model.classes]}')

    frame_rate = 1  # Definindo 1 FPS (intervalo de 1 segundo entre capturas)
    duration = 20  # Captura por 20 segundos
    start_time = time.time()

    with Progress(console=console) as progress:
        task = progress.add_task(
            '[cyan]Processando frames...[/cyan] [yellow](Pressione q para terminar)[/yellow]',
            total=duration,
        )

        while cap.isOpened():
            current_time = time.time()
            elapsed = current_time - start_time
            progress.update(task, completed=min(elapsed, duration))

            if elapsed >= duration:
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Controlando o FPS: espera até o próximo intervalo
            time.sleep(frame_rate)

            frame_count += 1
            results = model(frame, verbose=False)[0]
            person_detected = False
            best_confidence = 0
            best_pose = POSE_NAO_DETECTADA

            if annotated_frame_cv2:
                annotated_frame = results.plot()
                cv2.imshow('Sistema de Monitoramento de Poses - Deteccao em Tempo Real', annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    console.print('\n❌ Monitoramento interrompido pela janela do OpenCV.')
                    break

            for detection in results.boxes.data:
                class_id = int(detection[5])
                confidence = float(detection[4])

                if class_id in CLASSES_DETECTADAS:
                    person_detected = True
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_pose = CLASSES_DETECTADAS[class_id]

            detected_pose_this_frame = best_pose if person_detected else POSE_NAO_DETECTADA
            pose_durations[detected_pose_this_frame] += 1

    # Finalização e relatório
    cap.release()
    cv2.destroyAllWindows()
    end_time = time.time()
    total_time = end_time - start_time

    # Salvar CSV e JSON
    csv_path, json_path = salvar_csv_json_relatorio(pose_durations, total_time, output_dir, origem='video')
    console.print(f'💾 CSV salvo em: [bold]{os.path.abspath(csv_path)}[/]')
    console.print(f'💾 JSON salvo em: [bold]{os.path.abspath(json_path)}[/]')

    # Tentar Enviar JSON para o servidor
    try:
        enviar_json_para_servidor(json_path, server_url)
    except Exception as e:
        console.print(f'[bold red]❌ Erro ao tentar enviar o JSON para o servidor: {str(e)}[/]')

    return pose_durations


if __name__ == '__main__':
    try:
        config = load_config()  # Carregar configurações salvas
        initialize_app()
        if 'pyi_splash' in sys.modules:
            pyi_splash.close()

        # Perguntar apenas uma vez se deseja iniciar o monitoramento
        midia_path, weights_path, annotated_frame_cv2, output_dir, server_url = get_user_parameters(config)

        console.print('\n✨ Iniciando com as configurações:')
        console.print(f'📦 Fonte: {midia_path}')
        console.print(f'🎯 Arquivo de pesos: {weights_path}')
        console.print(f'📁 Diretório de saída: {output_dir}')
        console.print(f'🌐 Envio do JSON para: {server_url}')

        while True:  # Loop contínuo para o monitoramento
            _ = run_pose_monitoring(
                midia_path,
                weights_path,
                annotated_frame_cv2,
                output_dir,
                server_url
            )

    except KeyboardInterrupt:
        console.print('\n\n❌ Monitoramento interrompido pelo usuário', style='bold red')
    except Exception as e:
        console.print(f'\n\n❌ Erro inesperado: {str(e)}', style='bold red')
        console.print(f'[bold red]Detalhes do erro: {repr(e)}[/]')
