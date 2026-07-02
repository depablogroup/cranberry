project = "cranberry-rna"
author = "CRANBERRY developers"
release = "1.0.0a1"

extensions = ["myst_parser"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "dev/**"]
html_theme = "alabaster"
