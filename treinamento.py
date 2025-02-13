from ultralytics import YOLO
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

def train_yolov8_pose_model(data_yaml_path="data.yaml", pretrained_model="yolov11n.pt", epochs=100, imgsz=640):
    with console.status("[bold green]Inicializando treinamento...") as status:
        console.print(Panel.fit(
            """[bold blue]Configurações do Treinamento[/bold blue]
            Dataset YAML: {data_yaml_path}
            Modelo pré-treinado: {pretrained_model}
            Épocas: {epochs}
            Tamanho da imagem: {imgsz}""".format(
                data_yaml_path=data_yaml_path,
                pretrained_model=pretrained_model,
                epochs=epochs,
                imgsz=imgsz
            ),
            title="YOLOv8 Training",
            border_style="blue"
        ))

        try:
            model = YOLO(pretrained_model)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True
            ) as progress:
                progress.add_task("[cyan]Treinando modelo...", total=None)
                results = model.train(data=data_yaml_path, epochs=epochs, imgsz=imgsz)

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Informação", style="dim")
            table.add_column("Valor")
            
            table.add_row(
                "Diretório dos resultados",
                f"[green]{results.save_dir}[/green]"
            )
            table.add_row(
                "Arquivos de pesos salvos",
                "[blue]best.pt[/blue] e [blue]last.pt[/blue]"
            )
            table.add_row(
                "Logs e métricas",
                f"[yellow]{results.save_dir}/results.csv[/yellow]"
            )

            console.print("\n[bold green]✓ Treinamento concluído com sucesso![/bold green]")
            console.print(table)

            return results

        except Exception as e:
            console.print(Panel(
                f"[bold red]Erro durante o treinamento:[/bold red]\n{str(e)}\n\n"
                "[yellow]Verifique:[/yellow]\n"
                "► O arquivo 'data.yaml' existe e está no caminho correto\n"
                "► O dataset está formatado corretamente no formato YOLO\n"
                "► O modelo pré-treinado especificado é válido\n"
                "► As bibliotecas 'ultralytics', 'torch' e 'rich' estão instaladas",
                title="Erro",
                border_style="red"
            ))
            return None

if __name__ == "__main__":
    dataset_config_file = "data_poses.yaml"
    modelo_pre_treinado = "yolov11n.pt"
    num_epochs = 100
    tamanho_imagem = 640

    train_results = train_yolov8_pose_model(
        data_yaml_path=dataset_config_file,
        pretrained_model=modelo_pre_treinado,
        epochs=num_epochs,
        imgsz=tamanho_imagem
    )

    if train_results:
        console.print("\n[bold cyan]Resumo do Treinamento:[/bold cyan]")
        console.print(train_results)
        
        console.print(Panel(
            "[green]► Para validar o modelo: use o script 'val.py' ou a função 'model.val()'[/green]\n"
            "[blue]► Para usar o modelo: use o script 'predict.py' ou a função 'model.predict()'[/blue]\n"
            "[yellow]► Os pesos do modelo estão salvos no diretório indicado acima[/yellow]",
            title="Próximos Passos",
            border_style="cyan"
        ))
    else:
        console.print("[bold red]O treinamento falhou. Verifique as mensagens de erro acima.[/bold red]")