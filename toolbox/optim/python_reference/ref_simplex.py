"""Compare the SIMPLEX lines printed by test_simplex.m (MatLivre or MATLAB output, file given as the
first argument) with the HiGHS reference in ref_simplex.json."""
import json, re, sys
ref = json.load(open("ref_simplex.json")); got = {}
for l in open(sys.argv[1]):
    m = re.match(r"SIMPLEX\s+(\d+) ef=\s*(-?\d+) f=(\S+)", l)
    if m: got[int(m.group(1))] = (int(m.group(2)), float(m.group(3)))
bad = [r for r in ref if not (got[r["k"]][0] == r["status"] and (r["status"] != 1 or abs(got[r["k"]][1] - r["fun"]) <= 1e-6 * max(1, abs(r["fun"]))))]
print("mismatches with HiGHS:", len(bad), "of", len(ref))
for r in bad: print(r, got[r["k"]])
