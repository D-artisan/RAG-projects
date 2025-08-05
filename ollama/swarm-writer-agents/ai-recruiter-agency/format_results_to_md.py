import os
import ast
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR = RESULTS_DIR / "markdown"
OUTPUT_DIR.mkdir(exist_ok=True)

def dict_to_markdown(d, level=1):
    md = ""
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (dict, list)):
                md += f"{'#' * level} {k}\n\n"
                md += dict_to_markdown(v, level + 1)
            else:
                md += f"- **{k}**: {v}\n"
    elif isinstance(d, list):
        for i, item in enumerate(d, 1):
            md += f"{i}. {dict_to_markdown(item, level + 1)}"
    else:
        md += str(d) + "\n"
    return md

def process_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    try:
        # Try to parse as Python dict (single quotes)
        data = ast.literal_eval(content)
    except Exception:
        return None, None
    md = dict_to_markdown(data)
    return md, filepath.name

def main():
    for file in RESULTS_DIR.glob("analysis_*.txt"):
        md, name = process_file(file)
        if md:
            out_path = OUTPUT_DIR / (name.replace(".txt", ".md"))
            with open(out_path, "w") as out:
                out.write(md)
            print(f"Converted {file} -> {out_path}")

if __name__ == "__main__":
    main()
