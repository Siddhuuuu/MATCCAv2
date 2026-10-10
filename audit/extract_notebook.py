"""
extract_notebook.py - rebuild the v1 code that produced the paper from MatCCA.ipynb.

The GitHub .py files are older than the notebook's %%writefile cells. This writes the LAST version
of every %%writefile cell, plus the data helpers and shakespeare.txt, into OUT so the audit
scripts can run against exactly what produced Tables 1-3.

  python audit/extract_notebook.py /path/to/v1/MatCCA nbcode
  python -I audit/audit_tests.py nbcode
  python -I audit/allocator_math.py nbcode
"""
import json
import os
import shutil
import sys

repo, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
nb = json.load(open(os.path.join(repo, "MatCCA.ipynb")))
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if src.startswith("%%writefile"):
        first, rest = src.split("\n", 1)
        with open(os.path.join(out, os.path.basename(first.split()[1])), "w") as f:
            f.write(rest)
for name in ("real_data.py", "bpe_data.py", "shakespeare.txt"):
    shutil.copy(os.path.join(repo, name), out)
print("wrote", sorted(os.listdir(out)))
