import msvcrt  # Biblioteca para verificar se uma tecla foi pressionada
import time

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

console = Console()


def get_user_parameters():
    console.print(
        Panel('🎮 Configuração do Monitoramento', style='bold green')
    )

    # Seleção da fonte de vídeo
    video_source = Prompt.ask(
        '\n📹 Escolha a fonte de vídeo',
        choices=['webcam', 'arquivo'],
        default='webcam',
    )

    video_path = (
        '0'
        if video_source == 'webcam'
        else Prompt.ask(
            '🎥 Digite o caminho do arquivo de vídeo',
            default=ARQUIVO_VIDEO_PADRAO,
        )
    )

    # Seleção da duração
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
        else Prompt.ask(
            'Digite o caminho para o arquivo de pesos personalizado'
        )
    )

    # Seleção da visualização dos frames
    annotated_frame_cv2 = Confirm.ask(
        '\n🖼️ Visualizar frames com anotações?', default=True
    )

    return video_path, duration_seconds, weights_path, annotated_frame_cv2


def run_pose_monitoring(  # noqa: PLR0912, PLR0914, PLR0915
    video_path=ARQUIVO_VIDEO_PADRAO,
    duration_seconds=DURACAO_PADRAO,
    weights_path=ARQUIVO_PESOS,
    annotated_frame_cv2=True,
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
            model.classes = [0, 1, 2, 3]
        except Exception as e:
            console.print(
                f'[bold red]❌ Erro ao carregar o modelo:[/] {str(e)}'
            )
            return

    # Inicialização da câmera ou vídeo
    cap = cv2.VideoCapture(0 if video_path == '0' else video_path)
    if not cap.isOpened():
        console.print('[bold red]❌ Erro ao abrir fonte de vídeo!')
        return

    console.print('\n📊 Configurações:')
    console.print(f'- Duração planejada: {duration_seconds / 60:.1f} minutos')
    console.print(
        '- Classes detectadas:'
        + f' {[CLASSES_DETECTADAS[c] for c in model.classes]}'
    )

    # Processamento dos frames
    with Progress(console=console) as progress:
        task = progress.add_task(
            '[cyan]Processando frames...[/cyan] [yellow]'
            + '(Pressione q para terminar)[/yellow]',
            total=duration_seconds,
        )

        while cap.isOpened():
            current_time = time.time()
            elapsed = current_time - start_time

            # Atualiza a barra de progresso baseada no tempo real
            progress.update(task, completed=min(elapsed, duration_seconds))

            # Verifica se atingiu o tempo desejado
            if elapsed >= duration_seconds:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            results = model(frame, verbose=False)[
                0
            ]  # Obtém o primeiro resultado
            person_detected = False
            best_confidence = 0
            best_pose = POSE_NAO_DETECTADA

            if annotated_frame_cv2:
                # Adiciona as anotações ao frame
                annotated_frame = results.plot()

                # Mostra o frame com as anotações na tela
                cv2.imshow(
                    'Sistema de Monitoramento de '
                    + 'Poses - Deteccao em Tempo Real',
                    annotated_frame,
                )

                # tecla a ser pressionada na janela
                # do openvc para quebrar o loop
                key = cv2.waitKey(1) & 0xFF
                # Verifica se a tecla q foi pressionada na janela do OpenCV
                if key == ord('q'):
                    console.print(
                        '\n❌ Monitoramento interrompido'
                        + ' pela janela do OpenCV.'
                    )
                    break

            # Verifica se há tecla pressionada no terminal
            if msvcrt.kbhit():
                terminal_key = msvcrt.getch().decode().lower()
                if terminal_key == 'q':
                    console.print(
                        '\n❌ Monitoramento interrompido pelo terminal.'
                    )
                    break

            for detection in results.boxes.data:
                class_id = int(
                    detection[5]
                )  # O índice da classe está na posição 5
                confidence = float(
                    detection[4]
                )  # A confiança está na posição 4

                if class_id in CLASSES_DETECTADAS:
                    person_detected = True
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_pose = CLASSES_DETECTADAS[class_id]

            detected_pose_this_frame = (
                best_pose if person_detected else POSE_NAO_DETECTADA
            )
            pose_durations[detected_pose_this_frame] += 1

    # Finalização e relatório
    cap.release()
    cv2.destroyAllWindows()
    end_time = time.time()
    total_time = end_time - start_time

    # Criação do relatório em tabela
    table = Table(title='📊 Relatório de Monitoramento de Poses')
    table.add_column('Pose', style='cyan')
    table.add_column('Duração (min)', justify='right')
    table.add_column('Porcentagem', justify='right')

    # Calcular a proporção de tempo para cada pose
    total_frames = sum(pose_durations.values())
    for pose, frames in pose_durations.items():
        # Calcula a proporção do tempo total para cada pose
        duration_minutes = (frames / total_frames) * (total_time / 60)
        percentage = (frames / total_frames) * 100 if total_frames > 0 else 0
        table.add_row(pose, f'{duration_minutes:.2f}', f'{percentage:.1f}%')

    console.print('\n')
    console.print(table)
    console.print(
        '\n⏱️ Tempo total monitorado: [bold]'
        + f'{total_time / 60:.2f}[/] minutos'
    )
    console.print(
        f'⚡ Tempo de processamento: [bold]{total_time:.2f}[/] segundos'
    )

    return pose_durations


if __name__ == '__main__':
    try:
        while True:
            console.clear()
            console.print(
                Panel(
                    '🤖 Sistema de Monitoramento de Poses',
                    style='bold magenta',
                )
            )

            start_monitoring = Confirm.ask(
                '\n🚀 Iniciar novo monitoramento?', default=True
            )
            if not start_monitoring:
                console.print('\n👋 Até logo!', style='bold blue')
                break

            video_path, duration_seconds, weights_path, annotated_frame_cv2 = (
                get_user_parameters()
            )

            console.print('\n✨ Iniciando monitoramento com as configurações:')
            console.print(f'📹 Fonte de vídeo: {video_path}')
            console.print(f'⏱️ Duração: {duration_seconds / 60:.1f} minutos')
            console.print(f'🎯 Arquivo de pesos: {weights_path}')

            if Confirm.ask('\n▶️ Confirmar e começar?', default=True):
                report = run_pose_monitoring(
                    video_path,
                    duration_seconds,
                    weights_path,
                    annotated_frame_cv2,
                )

            if not Confirm.ask(
                '\n🔄 Deseja realizar outro monitoramento?', default=True
            ):
                console.print('\n👋 Até logo!', style='bold blue')
                break

    except KeyboardInterrupt:
        console.print(
            '\n\n❌ Monitoramento interrompido pelo usuário', style='bold red'
        )
    except Exception as e:
        console.print(f'\n\n❌ Erro inesperado: {str(e)}', style='bold red')
