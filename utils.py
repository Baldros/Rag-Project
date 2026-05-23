from pathlib import Path
from dotenv import load_dotenv
import os

def list_files(path:Path | str) -> list[Path | str]:
    """
    Função que lista arquivos:
    """

    path = Path(path)
    return list(path.iterdir())

def get_env_variables(env_path: Path | str | None = None) -> tuple[str, str]:
    """
    Função para carregar variáveis de ambiente a partir de um arquivo .env.
    """
    load_dotenv(dotenv_path=env_path)

    return os.getenv("OPENAI_API_KEY"), os.getenv("llm-model")