"""Generate the code reference pages and navigation."""

from pathlib import Path
from typing import List

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

source_paths = sorted(Path("./src/equation_tree/").rglob("*.py"))


special_paths: List[Path] = []  # [Path("./docs/docstrings.py")]

for path in source_paths:
    module_path = path.relative_to(".").with_suffix("")
    doc_path = path.relative_to(".").with_suffix(".md")

    full_doc_path = Path("reference", doc_path)


    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")
    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
