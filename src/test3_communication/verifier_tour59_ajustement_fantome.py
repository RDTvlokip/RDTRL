"""Tour 59 (28/09/2026) : ajustement de la loi du fantome sur la grille
eps x delta (lancer_tour59_grille_eps_delta.sh).

Pour chaque eps, les runs qui basculent donnent des temps t_i a delta_i.
Loi testee : t = t0 + kappa / sqrt(delta - delta_c'), soit
(t - t0)^-2 = A (delta - delta_c'),  A = 1 / kappa^2.
t0 est balaye (0 .. min(t)-1), la regression lineaire de (t-t0)^-2 sur
delta est faite pour chacun, on garde le t0 de R^2 maximal.
Predictions du carnet (28/09) : pente A = 30,2 (+-30 %), R^2 > 0,99,
delta_c'(1e-8) = pli + 3,9e-9, delta_c'(1e-7) = pli - 6,5e-9,
delta_c'(3e-8) = pli + 0,2e-9 (+-3e-9).
Sort aussi, par eps, le plus petit delta qui bascule et le plus grand qui
tient : l'encadrement direct, sans loi.
"""

import glob
import re
import numpy as np

PLI = 0.0134372100660973
KAPPA_PREDIT = 0.182


def lire(tag="grille"):
    runs = {}
    for f in glob.glob(f"D:/tmp/rdtrl_t59_{tag}_*.txt"):
        m = re.search(tag + r"_([0-9.e+-]+)_(-?[0-9.e+-]+)\.txt$", f)
        txt = open(f).read().strip()
        if not m or not txt:
            continue
        eps, off = m.group(1), float(m.group(2))
        pas = int(re.search(r"pas=(\d+)", txt).group(1))
        bas = int(re.search(r"bascule=(-?\d+)", txt).group(1))
        runs.setdefault(eps, []).append((off, bas, pas))
    return runs


def ajuster(off, t):
    best = None
    for t0 in range(-20000, int(t.min()), 5):
        y = (t - t0) ** -2.0
        A, B = np.polyfit(off, y, 1)
        pred = A * off + B
        ss = ((y - pred) ** 2).sum()
        tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss / tot if tot > 0 else float("nan")
        if best is None or r2 > best[0]:
            best = (r2, t0, A, -B / A)
    return best


if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "grille"
    for eps, lst in sorted(lire(tag).items(), key=lambda kv: float(kv[0])):
        lst.sort()
        casse = [(o, b) for o, b, _ in lst if b >= 0]
        tient = [(o, p) for o, b, p in lst if b < 0]
        print(f"=== eps={eps} : {len(lst)} runs, {len(casse)} basculent")
        print("    " + "  ".join(f"{o:+.0e}:{'B' + str(b) if b >= 0 else 'tient'}" for o, b, _ in lst))
        if casse and tient:
            print(f"    encadrement direct : tient jusqu'a {max(o for o, _ in tient if o < min(c for c, _ in casse)) if any(o < min(c for c, _ in casse) for o, _ in tient) else float('nan'):+.1e}, "
                  f"premier qui casse {min(c for c, _ in casse):+.1e}")
        if len(casse) >= 3:  # 3 points = 3 parametres libres : R2 = 1 par construction, a lire comme une simple inversion
            off = np.array([o for o, _ in casse]); t = np.array([b for _, b in casse], dtype=float)
            r2, t0, A, dc = ajuster(off, t)
            print(f"    ajustement : t0={t0}  A={A:.1f} (kappa={1 / np.sqrt(A):.3f}, predit A=30,2)  delta_c'-pli={dc:+.2e}  R2={r2:.4f}")
