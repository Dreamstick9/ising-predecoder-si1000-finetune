import json, sys, math, re
# Print a compact table from "[Inference Summary]" lines. usage: summarize.py <shots_per_basis> <log>...
shots = int(sys.argv[1])
print(f"{'run':<28}{'PyMatching':>16}{'with pre-decoder':>18}{'improvement':>13}{'+/-':>8}")
for path in sys.argv[2:]:
    name = path.split('/')[-1].replace('.log','')
    try:
        line = [l for l in open(path) if 'Inference Summary' in l][-1]
    except (IndexError, FileNotFoundError):
        print(f"{name:<28} (no result yet)"); continue
    d = json.loads(line[line.index('{'):])['ler']
    a, b = d['avg_no_predecoder'], d['avg_after_predecoder']
    n = 2 * shots
    se_a, se_b = math.sqrt(a*(1-a)/n), math.sqrt(b*(1-b)/n)
    ratio = a / b if b > 0 else float('nan')
    se_ratio = ratio * math.sqrt((se_a/a)**2 + (se_b/b)**2) if a > 0 and b > 0 else float('nan')
    print(f"{name:<28}{a:>16.3e}{b:>18.3e}{ratio:>12.3f}x{se_ratio:>8.3f}")
