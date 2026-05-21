from pathlib import Path

def list_files(path:Path | str) -> list[Path | str]:
    """
    Função que lista arquivos:
    """

    path = Path(path)
    return list(path.iterdir())

if __name__ == "__main__":
    print(list_files("E:\Estudo\Fisica\Fisica III"))