"""One-time dependency migration after the dated archive was created."""

import ast
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archives/2026-10-07"
manifest = json.loads((ARCHIVE / "manifest.json").read_text(encoding="utf-8-sig"))
assert len((ARCHIVE / "completed.txt").read_text(encoding="utf-8-sig").splitlines()) == len(manifest)


def edit(path, transform):
    source = ROOT / path
    target = ARCHIVE / "compatibilite" / path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    manifest.append({"original": path, "archived": target.relative_to(ARCHIVE).as_posix(),
                     "group": "compatibility_backup", "bytes": source.stat().st_size,
                     "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "action": "copy"})
    source.write_text(transform(source.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")


old_data = (ARCHIVE / "historique/src/experiments/comparison/comparison_data.py").read_text(encoding="utf-8")
node = next(n for n in ast.parse(old_data).body if isinstance(n, ast.FunctionDef) and n.name == "pack_sequences")
helper = ast.get_source_segment(old_data, node)
edit("src/data/batching.py", lambda s: s + "\n\n" + helper + "\n")
edit("tests/test_benchmark_tcn.py", lambda s: s.replace(
    "from src.data.batching import batches, collate", "from src.data.batching import batches, collate, pack_sequences"
).replace("from src.experiments.comparison.comparison_data import pack_sequences\n", ""))
retired = {Path(i["original"]).name for i in manifest if i["group"] == "old_feature"}
edit("src/provenance.py", lambda s: "\n".join(line for line in s.splitlines()
     if not any(line.lstrip().startswith('"' + name + '":') for name in retired)) + "\n")


def clean_runner(s):
    start, end = s.index("def codebert_stage("), s.index("def tcn_stage(")
    s = s[:start] + s[end:]
    s = s.replace('    "codebert": "CodeBERT fine-tuné (archive)",\n', "")
    s = s.replace('    if "codebert" in config["models"]:\n        source_paths.append(source_path("benchmark_codebert.py"))\n', "")
    s = s.replace('            "codebert-pilot",\n            "codebert",\n', "")
    start = s.index('    if args.stage in ("codebert-pilot", "codebert")')
    end = s.index('    summary = write_report', start)
    s = s[:start] + s[end:]
    marker = '    if Path(run_id).name != run_id'
    s = s.replace(marker, '    if set(config["models"]) - set(NAMES):\n        raise ValueError("Historical models must be restored in a separate workspace")\n' + marker)
    return s


edit("src/experiments/benchmark/run_model_benchmark.py", clean_runner)
edit("src/visualization/plot_results.py", lambda s: s.replace("présentés dans results/comparison.", "présentés dans results/benchmark/cnn-xgboost-tcn-20261005."))
for path in (".gitignore", ".gitattributes", "pyproject.toml", "README.md", "docs/experiments.md", "docs/architecture.md", "docs/memoire/README.md"):
    def transform(s, path=path):
        if path == ".gitignore":
            return s + "\n# Local caches retained for recovery, never publish them.\n/archives/*/local/\n"
        if path == ".gitattributes":
            return s + "\n# Archived evidence must retain its byte-level hashes.\narchives/** -text whitespace=cr-at-eol\narchives/**/*.svg -whitespace\n"
        if path == "pyproject.toml":
            return s.replace('"src/preprocessing.py",', '"archives", "src/preprocessing.py",')
        if path == "README.md":
            return s + "\nLes expériences anciennes, brouillons et caches ont été rangés dans les [archives du 7 octobre 2026](archives/2026-10-07/README.md). Les résultats finaux restent en place.\n"
        # Rewrite links to moved files or whole archived historical trees.
        def link(m):
            link_path = m.group(1)
            if "://" in link_path or link_path.startswith("#"):
                return m.group(0)
            resolved = (ROOT / path).parent.joinpath(link_path).resolve()
            try:
                relative = resolved.relative_to(ROOT).as_posix()
            except ValueError:
                return m.group(0)
            archived = ARCHIVE / "historique" / relative
            if archived.exists():
                import os
                return "](" + os.path.relpath(archived, (ROOT / path).parent).replace("\\", "/") + ")"
            return m.group(0)
        s = re.sub(r"\]\(([^)]+)\)", link, s)
        if path == "docs/experiments.md":
            start = s.index("La commande `python -u -m src.experiments.comparison.run_comparison")
            end = s.index("Comparaison terminée", start)
            s = s[:start] + "Le code, la configuration et les résultats de cette expérience sont archivés dans `archives/2026-10-07/historique/`. Pour les réexécuter, restaurer les chemins d'origine dans une copie séparée du projet suivant le README de l'archive.\n\n" + s[end:]
        if path == "docs/architecture.md":
            s = "\n".join(line for line in s.splitlines() if not any(x in line for x in ("| `python -m src.run_comparison`", "| `python -m src.plot_comparison`"))) + "\n"
        s += "\nArchivage du 7 octobre 2026 : les scripts de comparaison ancienne et de migration documentaire sont conservés dans `archives/2026-10-07/historique/`, sous leurs chemins d'origine. Les commandes historiques nécessitent une restauration dans une copie séparée ; voir le [guide de restauration](" + ("../../" if path.startswith("docs/memoire/") else "../") + "archives/2026-10-07/README.md).\n"
        return s
    edit(path, transform)
(ARCHIVE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"archived": len(manifest), "moved": sum(i["action"] == "move" for i in manifest), "bytes": sum(i["bytes"] for i in manifest)}))
