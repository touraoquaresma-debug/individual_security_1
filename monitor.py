import os
import sys
import time
import csv

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


def get_user_parameters():
    console.print(
        Panel('🎮 Configuração do Monitoramento', style='bold green')
    )

    # Seleção da fonte
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

    # Seleção da duração (apenas para webcam/vídeo)
    duration_seconds = DURACAO_PADRAO
    if video_source in ('webcam', 'video'):
        duration_str = Prompt.ask(
            '\n⏱️ Digite a duração desejada em minutos', default='5'
        )
        duration_seconds = float(duration_str) * 60

    # Seleção do arquivo de pesos
    use_default_weights = Confirm.ask(
        '\n🎯 Usar arquivo de pesos padrão?', default=True
    )
    weights_path = (
        ARQUIVO_PESOS
        if use_default_weights
        else Prompt.ask('Digite o caminho para o arquivo de pesos personalizado')
    )

    # Visualização
    annotated_frame_cv2 = Confirm.ask(
        '\n🖼️ Visualizar frames com anotações?', default=True
    )

    # Diretório de saída para CSV
    output_dir = Prompt.ask(
        '\n📁 Diretório para salvar o CSV',
        default=os.path.join('.', 'relatorios'),
    )

    return midia_path, duration_seconds, weights_path, annotated_frame_cv2, output_dir


def eh_imagem(caminho: str) -> bool:
    ext = os.path.splitext(caminho)[1].lower()
    return ext in ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


def salvar_csv_relatorio(pose_durations, total_time, csv_dir, origem='midia'):
    """
    Salva a tabela final como CSV no diretório informado.
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
        linhas.append([pose, f'{duracao_min:.2f}', f'{porcentagem:.1f}%'])

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    nome_arquivo = f'relatorio_poses_{origem}_{timestamp}.csv'
    caminho_csv = os.path.join(csv_dir, nome_arquivo)

    with open(caminho_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['pose', 'duracao_min', 'porcentagem'])
        writer.writerows(linhas)

    return caminho_csv


def run_pose_monitoring(  # noqa: PLR0912, PLR0914, PLR0915
    midia_path=ARQUIVO_VIDEO_PADRAO,
    duration_seconds=DURACAO_PADRAO,
    weights_path=ARQUIVO_PESOS,
    annotated_frame_cv2=True,
    output_dir='./relatorios',
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
            # limite às classes de interesse (ajuste conforme seu rótulo)
            model.classes = [0, 1, 2, 3]
            # confiança mínima (em versões recentes usa-se no predict)
            # manter para compatibilidade com seu código
            model.conf = 0.5
        except Exception as e:
            console.print(f'[bold red]❌ Erro ao carregar o modelo:[/] {str(e)}')
            return

    # Modo IMAGEM
    if midia_path != '0' and eh_imagem(midia_path):
        console.print('\n📊 Configurações:')
        console.print('- Modo: imagem única')
        console.print(f'- Arquivo: {midia_path}')
        console.print(
            '- Classes detectadas:'
            + f' {[CLASSES_DETECTADAS[c] for c in model.classes]}'
        )

        # Verificando se o caminho da imagem está correto e se o arquivo existe
        if not os.path.exists(midia_path):
            console.print(f'[bold red]❌ Erro: O arquivo de imagem não foi encontrado no caminho "{midia_path}".[/]')
            return

        # Tentativa de abrir a imagem
        img = cv2.imread(midia_path)
        if img is None:
            console.print(f'[bold red]❌ Erro ao abrir a imagem! Verifique se o formato é suportado (jpg, png, etc.).[/]')
            return

        # Inferência
        results = model(img, verbose=False)[0]
        person_detected = False
        best_confidence = 0
        best_pose = POSE_NAO_DETECTADA

        if annotated_frame_cv2:
            annotated_frame = results.plot()
            cv2.imshow('Sistema de Monitoramento de Poses - Imagem', annotated_frame)
            # fecha ao pressionar qualquer tecla ou após 2s
            if cv2.waitKey(2000) & 0xFF == ord('q'):
                pass
            cv2.destroyAllWindows()

        # Loop sobre detecções
        for detection in results.boxes.data:
            class_id = int(detection[5])  # índice da classe
            confidence = float(detection[4])  # confiança
            if class_id in CLASSES_DETECTADAS:
                person_detected = True
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_pose = CLASSES_DETECTADAS[class_id]

        detected_pose_this_frame = best_pose if person_detected else POSE_NAO_DETECTADA
        pose_durations[detected_pose_this_frame] += 1
        frame_count = 1

        # tempo total ~0 para imagem (não é sequência)
        end_time = time.time()
        total_time = end_time - start_time

        # Tabela (console)
        table = Table(title='📊 Relatório de Monitoramento de Poses (Imagem)')
        table.add_column('Pose', style='cyan')
        table.add_column('Duração (min)', justify='right')
        table.add_column('Porcentagem', justify='right')

        total_frames = max(sum(pose_durations.values()), 1)
        for pose, frames in pose_durations.items():
            duration_minutes = (frames / total_frames) * (total_time / 60) if total_time > 0 else 0
            percentage = (frames / total_frames) * 100
            table.add_row(pose, f'{duration_minutes:.2f}', f'{percentage:.1f}%')

        console.print('\n')
        console.print(table)
        console.print(
            '\n⏱️ Tempo total processado: [bold]'
            + f'{total_time:.2f}[/] segundos'
        )

        # CSV
        csv_path = salvar_csv_relatorio(
            pose_durations, total_time, output_dir, origem='imagem'
        )
        console.print(f'💾 CSV salvo em: [bold]{os.path.abspath(csv_path)}[/]')

        return pose_durations

    # Modo WEBCAM/VÍDEO
    cap = cv2.VideoCapture(0 if midia_path == '0' else midia_path)
    if not cap.isOpened():
        console.print('[bold red]❌ Erro ao abrir fonte de vídeo!')
        return

    console.print('\n📊 Configurações:')
    console.print(f'- Duração planejada: {duration_seconds / 60:.1f} minutos')
    console.print(
        '- Classes detectadas:'
        + f' {[CLASSES_DETECTADAS[c] for c in model.classes]}'
    )

    with Progress(console=console) as progress:
        task = progress.add_task(
            '[cyan]Processando frames...[/cyan] [yellow](Pressione q para terminar)[/yellow]',
            total=duration_seconds,
        )

        while cap.isOpened():
            current_time = time.time()
            elapsed = current_time - start_time
            progress.update(task, completed=min(elapsed, duration_seconds))

            if elapsed >= duration_seconds:
                break

            ret, frame = cap.read()
            if not ret:
                break

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

            # Verifica se há tecla pressionada no terminal (apenas Windows)
            if msvcrt and msvcrt.kbhit():
                terminal_key = msvcrt.getch().decode(errors='ignore').lower()
                if terminal_key == 'q':
                    console.print('\n❌ Monitoramento interrompido pelo terminal.')
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

    # Tabela (console)
    table = Table(title='📊 Relatório de Monitoramento de Poses')
    table.add_column('Pose', style='cyan')
    table.add_column('Duração (min)', justify='right')
    table.add_column('Porcentagem', justify='right')

    total_frames = max(sum(pose_durations.values()), 1)
    for pose, frames in pose_durations.items():
        duration_minutes = (frames / total_frames) * (total_time / 60)
        percentage = (frames / total_frames) * 100
        table.add_row(pose, f'{duration_minutes:.2f}', f'{percentage:.1f}%')

    console.print('\n')
    console.print(table)
    console.print('\n⏱️ Tempo total monitorado: [bold]' + f'{total_time / 60:.2f}[/] minutos')
    console.print(f'⚡ Tempo de processamento: [bold]{total_time:.2f}[/] segundos')

    # CSV
    origem = 'webcam' if midia_path == '0' else 'video'
    csv_path = salvar_csv_relatorio(pose_durations, total_time, output_dir, origem=origem)
    console.print(f'💾 CSV salvo em: [bold]{os.path.abspath(csv_path)}[/]')

    return pose_durations


if __name__ == '__main__':
    try:
        initialize_app()
        if 'pyi_splash' in sys.modules:
            pyi_splash.close()
        while True:
            console.clear()
            console.print(
                Panel('🤖 Sistema de Monitoramento de Poses', style='bold magenta')
            )

            start_monitoring = Confirm.ask('\n🚀 Iniciar novo monitoramento?', default=True)
            if not start_monitoring:
                console.print('\n👋 Até logo!', style='bold blue')
                break

            midia_path, duration_seconds, weights_path, annotated_frame_cv2, output_dir = (
                get_user_parameters()
            )

            console.print('\n✨ Iniciando com as configurações:')
            console.print(f'📦 Fonte: {midia_path}')
            if midia_path in ('0', ARQUIVO_VIDEO_PADRAO) or not eh_imagem(midia_path):
                console.print(f'⏱️ Duração: {duration_seconds / 60:.1f} minutos')
            console.print(f'🎯 Arquivo de pesos: {weights_path}')
            console.print(f'📁 Diretório de saída: {output_dir}')

            if Confirm.ask('\n▶️ Confirmar e começar?', default=True):
                _ = run_pose_monitoring(
                    midia_path,
                    duration_seconds,
                    weights_path,
                    annotated_frame_cv2,
                    output_dir,
                )

            if not Confirm.ask('\n🔄 Deseja realizar outro monitoramento?', default=True):
                console.print('\n👋 Até logo!', style='bold blue')
                break

    except KeyboardInterrupt:
        console.print('\n\n❌ Monitoramento interrompido pelo usuário', style='bold red')
    except Exception as e:
        console.print(f'\n\n❌ Erro inesperado: {str(e)}', style='bold red')
        console.print(f'[bold red]Detalhes do erro: {repr(e)}[/]')
