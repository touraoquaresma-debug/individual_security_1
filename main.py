import cv2
import time
from ultralytics import YOLO
from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from constants import *

console = Console()

def get_user_parameters():
    console.print(Panel("🎮 Configuração do Monitoramento", style="bold green"))
    
    # Seleção da fonte de vídeo
    video_source = Prompt.ask(
        "\n📹 Escolha a fonte de vídeo",
        choices=["webcam", "arquivo"],
        default="webcam"
    )
    
    video_path = "0" if video_source == "webcam" else Prompt.ask(
        "🎥 Digite o caminho do arquivo de vídeo",
        default=ARQUIVO_VIDEO_PADRAO
    )
    
    # Seleção da duração
    duration_str = Prompt.ask(
        "\n⏱️ Digite a duração desejada em minutos",
        default="5"
    )
    duration_seconds = float(duration_str) * 60
    
    # Seleção do arquivo de pesos
    use_default_weights = Confirm.ask(
        "\n🎯 Usar arquivo de pesos padrão?",
        default=True
    )
    weights_path = ARQUIVO_PESOS if use_default_weights else Prompt.ask(
        "Digite o caminho para o arquivo de pesos personalizado"
    )
    
    return video_path, duration_seconds, weights_path

def run_pose_monitoring(video_path=ARQUIVO_VIDEO_PADRAO, duration_seconds=DURACAO_PADRAO, weights_path=ARQUIVO_PESOS):
    # Inicialização do monitoramento
    console.print(Panel("🎥 Sistema de Monitoramento de Poses", style="bold blue"))
    
    start_time = time.time()
    pose_durations = {
        "em pé": 0, 
        "sentado": 0, 
        "deitado": 0, 
        POSE_NAO_DETECTADA: 0
    }
    frame_count = 0
    detected_pose_last_frame = POSE_NAO_DETECTADA

    # Carregamento do modelo YOLO
    with console.status("[bold green]Carregando o modelo YOLO..."):
        try:
            model = YOLO(weights_path)
            model.classes = [0, 1, 2]
        except Exception as e:
            console.print(f"[bold red]❌ Erro ao carregar o modelo:[/] {str(e)}")
            return

    # Inicialização da câmera ou vídeo
    cap = cv2.VideoCapture(0 if video_path == "0" else video_path)
    if not cap.isOpened():
        console.print("[bold red]❌ Erro ao abrir fonte de vídeo!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_PADRAO
    target_frame_count = int(duration_seconds * fps)

    console.print(f"\n📊 Configurações:")
    console.print(f"- Duração planejada: {duration_seconds/60:.1f} minutos")
    console.print(f"- Classes detectadas: {[CLASSES_DETECTADAS[c] for c in model.classes]}")

    # Processamento dos frames
    with Progress() as progress:
        task = progress.add_task("[cyan]Processando frames...", total=target_frame_count)
        
        while cap.isOpened() and frame_count < target_frame_count:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress.update(task, advance=1)

            results = model(frame)[0]  # Obtém o primeiro resultado
            
            person_detected = False
            best_confidence = 0
            best_pose = POSE_NAO_DETECTADA

            for detection in results.boxes.data:
                class_id = int(detection[5])  # O índice da classe está na posição 5
                confidence = float(detection[4])  # A confiança está na posição 4
                
                if class_id in CLASSES_DETECTADAS:
                    person_detected = True
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_pose = CLASSES_DETECTADAS[class_id]

            detected_pose_this_frame = best_pose if person_detected else POSE_NAO_DETECTADA
            pose_durations[detected_pose_this_frame] += 1
            detected_pose_last_frame = detected_pose_this_frame

    # Finalização e relatório
    cap.release()
    cv2.destroyAllWindows()
    elapsed_time = time.time() - start_time

    total_frames_processed = frame_count
    total_duration_processed_seconds = total_frames_processed / fps
    total_duration_processed_minutes = total_duration_processed_seconds / 60

    # Criação do relatório em tabela
    table = Table(title="📊 Relatório de Monitoramento de Poses")
    table.add_column("Pose", style="cyan")
    table.add_column("Duração (min)", justify="right")
    table.add_column("Porcentagem", justify="right")

    for pose, frames in pose_durations.items():
        duration_seconds = frames / fps
        duration_minutes = duration_seconds / 60
        percentage = (duration_seconds / total_duration_processed_seconds) * 100 if total_duration_processed_seconds > 0 else 0
        table.add_row(
            pose,
            f"{duration_minutes:.2f}",
            f"{percentage:.1f}%"
        )

    console.print("\n")
    console.print(table)
    console.print(f"\n⏱️ Tempo total monitorado: [bold]{total_duration_processed_minutes:.2f}[/] minutos")
    console.print(f"⚡ Tempo de processamento: [bold]{elapsed_time:.2f}[/] segundos")

    return pose_durations


if __name__ == "__main__":
    try:
        while True:
            console.clear()
            console.print(Panel("🤖 Sistema de Monitoramento de Poses", style="bold magenta"))
            
            start_monitoring = Confirm.ask("\n🚀 Iniciar novo monitoramento?", default=True)
            if not start_monitoring:
                console.print("\n👋 Até logo!", style="bold blue")
                break
                
            video_path, duration_seconds, weights_path = get_user_parameters()
            
            console.print("\n✨ Iniciando monitoramento com as configurações:")
            console.print(f"📹 Fonte de vídeo: {video_path}")
            console.print(f"⏱️ Duração: {duration_seconds/60:.1f} minutos")
            console.print(f"🎯 Arquivo de pesos: {weights_path}")
            
            if Confirm.ask("\n▶️ Confirmar e começar?", default=True):
                report = run_pose_monitoring(video_path, duration_seconds, weights_path)
            
            if not Confirm.ask("\n🔄 Deseja realizar outro monitoramento?", default=True):
                console.print("\n👋 Até logo!", style="bold blue")
                break
                
    except KeyboardInterrupt:
        console.print("\n\n❌ Monitoramento interrompido pelo usuário", style="bold red")
    except Exception as e:
        console.print(f"\n\n❌ Erro inesperado: {str(e)}", style="bold red")