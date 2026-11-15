"""Side-by-side comparison of reference_matlab.txt and reference_matlivre.txt (one line per test case)."""
import re, sys
def load(fn):
    d = {}
    for l in open(fn, encoding="utf-8", errors="replace"):
        m = re.match(r"(RESULT|LAMBDA|HS|GLOBAL|CHECKGRAD|OUTFCN|MILP|SHAPE|CHECK|SIMPLEX)\s+(\S+)\s+(.*)", l.strip())
        if m:
            body = re.sub(r"\s*t=[\d.]+s|time=[\d.]+s|,\s*[\d.]+s\)", "", m.group(3))
            d[(m.group(1), m.group(2))] = body
    return d
a = load(sys.argv[1] if len(sys.argv) > 1 else "reference_matlab.txt")
b = load(sys.argv[2] if len(sys.argv) > 2 else "reference_matlivre.txt")
same = diff = 0
for k in a:
    if k[0] == "SIMPLEX": continue
    x, y = a[k], b.get(k, "(missing)")
    if x == y: same += 1; continue
    diff += 1
    print(f"{k[0]} {k[1]}\n   MATLAB   {x}\n   MatLivre {y}")
print(f"\n{same} identical lines, {diff} different (SIMPLEX lines checked by ref_simplex.py)")
