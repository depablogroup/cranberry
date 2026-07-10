from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_EXT = Path(__file__).resolve().parent / '_ext'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DOCS_EXT))

project = 'cranberry-rna'
author = 'CRANBERRY developers'
release = '1.0.0a1'

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'argparse_help',
]
autosummary_generate = True
autodoc_typehints = 'description'
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'dev/**']
html_theme = 'alabaster'
