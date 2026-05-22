"""
Ponto de entrada para indexação de PDFs.

Toda a lógica está no pacote `processing/`.
Este arquivo existe apenas para manter compatibilidade
com o comando já usado no terminal.
"""

from processing import index_folder
from pathlib import Path

if __name__ == "__main__":
    PDF_DIR = Path(input("Digite o caminho da pasta com os PDFs: "))
    index_folder(PDF_DIR)