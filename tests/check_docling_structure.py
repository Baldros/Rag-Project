"""Integration with Docling's real hierarchy; no model weights are required.

Run separately: python tests/check_docling_structure.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import patch
from docling_core.types.doc import DoclingDocument, DocItemLabel, Size, BoundingBox, ProvenanceItem
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from kb.ingest import segment


def provenance(page):
    return ProvenanceItem(page_no=page, charspan=(0, 10),
                          bbox=BoundingBox(l=10, t=20, r=100, b=40))


first = DoclingDocument(name="part1")
for page in range(1, 16):
    first.add_page(page_no=page, size=Size(width=600, height=800))
first.add_heading(text="2.2 Estatísticas suficientes", level=1, prov=provenance(15))
first.add_text(label=DocItemLabel.TEXT, text="Começo da seção", prov=provenance(15))
second = DoclingDocument(name="part2")
second.add_page(page_no=16, size=Size(width=600, height=800))
second.add_page(page_no=17, size=Size(width=600, height=800))  # blank page retained
second.add_text(label=DocItemLabel.TEXT, text="Continuação sem novo título", prov=provenance(16))
with patch.object(segment, "iter_blocks", return_value=[(1, 15, first), (16, 17, second)]), \
     patch.object(segment, "get_chunker", return_value=HierarchicalChunker()):
    sections, chunks = segment.segment_document("fixture", {})
assert len(sections) == 1, sections
assert sections[0].heading == "2.2 Estatísticas suficientes"
assert sections[0].page_start == 15 and sections[0].page_end == 16
assert "Continuação" in sections[0].text
assert all(c.heading_path == sections[0].heading_path for c in chunks)
print("Docling integration passed: heading spans blocks; original pages retained")
