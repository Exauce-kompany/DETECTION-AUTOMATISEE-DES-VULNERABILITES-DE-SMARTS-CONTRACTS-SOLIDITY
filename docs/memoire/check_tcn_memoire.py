"""Validate the staged/final TCN thesis against pinned benchmark outputs."""
from pathlib import Path
import argparse
import hashlib
import json
import re

from docx import Document
import pdfplumber
from pdf2image import convert_from_path
from PIL import ImageChops

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=Path, default=HERE / 'tcn_staging')
parser.add_argument('--docx', type=Path)
parser.add_argument('--pdf', type=Path)
parser.add_argument('--require-complete', action='store_true')
args = parser.parse_args()
stage = args.stage
docx = args.docx or stage / 'draft_documents/Memoire_SMART_BUG_Exauce_Kompani.docx'
pdf = args.pdf or stage / 'render/Memoire_SMART_BUG_Exauce_Kompani.pdf'
document = Document(docx)
source = (stage / 'content.md').read_text(encoding='utf-8')
before = (HERE / 'before_tcn_models/content.md').read_text(encoding='utf-8')
snapshot = json.loads((stage / 'comparison_snapshot.json').read_text(encoding='utf-8'))
index = json.loads((stage / 'document_index.json').read_text(encoding='utf-8'))
summary_path = ROOT / snapshot['summary_source']
summary = json.loads(summary_path.read_text(encoding='utf-8'))
assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == snapshot['summary_sha256_at_import']
if snapshot.get('runtime_notes_source'):
    assert hashlib.sha256((ROOT / snapshot['runtime_notes_source']).read_bytes()).hexdigest() == snapshot['runtime_notes_sha256'], 'Runtime provenance changed after import'
if args.require_complete:
    assert summary['complete'] and snapshot['tcn_metrics']
def acknowledgment(s):
    return s[s.index('@PAGE ii\n'):s.index('@PAGE iv\n')]
assert acknowledgment(before) == acknowledgment(source)
def clean(text):
    text = text.replace('Christian-\nNoah', 'Christian-Noah')
    text = re.sub(r'(?<=\w)-\n(?=\w)', '', text)
    return re.sub(r'\s+', ' ', text).strip()
reference = json.loads((HERE / 'pages.json').read_text(encoding='utf-8'))
with pdfplumber.open(pdf) as reader:
    pages = reader.pages
    assert len(pages) == 44
    labels = re.findall(r'^@PAGE (.+)$', source, re.M)[1:]
    assert len(labels) == 43
    for page, label in zip(pages[1:], labels):
        assert page.extract_text().splitlines()[-1] == label
        assert not any(c['x0'] < 60 or c['x1'] > 545 for c in page.chars), (label, 'horizontal overflow')
    original = ' '.join('\n'.join(t.splitlines()[1:-1]) for t in reference[2:4])
    current = ' '.join('\n'.join(p.extract_text().splitlines()[1:-1]) for p in pages[2:4])
    assert clean(original) == clean(current), 'Acknowledgments differ from retained reference'
    full = '\n'.join(page.extract_text() for page in pages)
    assert 'CodeBERT' not in full and 'CodeT5' not in full
    assert all(name in full for name in ('CNN', 'XGBoost', 'TCN', 'GOPALI', 'BAI'))
    assert not re.search(r'\bV3\b', full, re.I)
    assert '84,54' not in full
    for title, label in index['figures']:
        page = pages[10 + int(label) - 1].extract_text()
        assert 'Figure ' + title.split()[0] in page, (title, label)
    for title, label in index['tables']:
        page = pages[10 + int(label) - 1].extract_text()
        assert 'Tableau ' + title.split()[0] in page, (title, label)
assert len(document.inline_shapes) == 15 and len(document.tables) == 8
def pct(value):
    return f'{100 * value:.2f}'.replace('.', ',')
rows = [[c.text for c in row.cells] for row in document.tables[-1].rows]
keys = ('f1_macro', 'recall_vulnerable', 'false_positive_rate', 'average_precision')
for row, name in zip(rows[1:], ('cnn_bilstm', 'xgboost', 'tcn')):
    metrics = summary['models'][name]['metrics']
    if metrics:
        assert row[1:] == [pct(metrics[k]) for k in keys], (name, row)
    else:
        assert row == ['TCN (en préparation)'] + ['Calcul à compléter'] * 4
assert summary['models']['xgboost']['selected']['seed'] == 101
assert summary['models']['xgboost']['metrics'] == snapshot['metrics']
# Pixel equality is checked using the same Poppler, resolution and PDF inputs.
poppler = 'C:/Users/Exauce/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'
old_images = convert_from_path(str(HERE / 'before_tcn_models/Memoire_SMART_BUG_Exauce_Kompani.pdf'), dpi=171, first_page=3, last_page=4, poppler_path=poppler)
new_images = convert_from_path(str(pdf), dpi=171, first_page=3, last_page=4, poppler_path=poppler)
assert len(old_images) == len(new_images) == 2
pixel_same = [ImageChops.difference(old.convert('RGB'), new.convert('RGB')).getbbox() is None for old, new in zip(old_images, new_images)]
assert all(pixel_same), 'Acknowledgments visual pixels differ'
report = {'pages': 44, 'figures': 15, 'tables': 8, 'acknowledgments_source_exact': True, 'acknowledgments_reference_text_match': True, 'acknowledgments_pixel_identical': pixel_same, 'pagination_checked': True, 'caption_page_references_checked': True, 'bibliography_entries': index['bibliography_entries'], 'abbreviations': index['abbreviations'], 'source_words': index['words'], 'comparison_status': snapshot['status'], 'benchmark_run_id': snapshot['run_id'], 'comparison_table_matches_saved_summary': True, 'summary_sha256': snapshot['summary_sha256_at_import'], 'xgboost_selected_seed': 101, 'tcn_selected_seed': (snapshot.get('tcn_selected') or {}).get('seed'), 'docx_sha256': hashlib.sha256(docx.read_bytes()).hexdigest(), 'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest()}
report['runtime_notes_sha256'] = snapshot.get('runtime_notes_sha256')
(stage / 'qa.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
