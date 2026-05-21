"""
Ponto de entrada para indexação de PDFs.

Toda a lógica está no pacote `processing/`.
Este arquivo existe apenas para manter compatibilidade
com o comando já usado no terminal.
"""

from processing import index_folder
from processing.config import PDF_DIR

if __name__ == "__main__":
    index_folder(PDF_DIR)