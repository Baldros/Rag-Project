from pathlib import Path
from docling.document_converter import DocumentConverter

pdf_path = Path(r"E:\Estudo\Fisica\Fisica III\Fisica_3_Sears_14a_ed.pdf")
converter = DocumentConverter()

# Define o tamanho do bloco e o total de páginas do livro (ajuste o total)
tamanho_bloco = 50
total_paginas = 500 

markdown_completo = ""

for pagina_inicial in range(1, total_paginas + 1, tamanho_bloco):
    pagina_final = min(pagina_inicial + tamanho_bloco - 1, total_paginas)
    print(f"Convertendo páginas {pagina_inicial} até {pagina_final}...")
    
    result = converter.convert(pdf_path, page_range=(pagina_inicial, pagina_final))
    markdown_completo += result.document.export_to_markdown() + "\n\n"

Path("out").mkdir(exist_ok=True)
Path("out/Fisica_3_Completo.md").write_text(markdown_completo, encoding="utf-8")

print("Conversão finalizada com sucesso!")
