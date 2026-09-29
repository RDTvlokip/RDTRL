"""Tour 59 (29/09/2026) : ou le regime lisse (loi du fantome deterministe) devient-il
stochastique quand eps diminue ? Indicateur : dispersion des temps de bascule
sur 10 phases k largement ecartees (k = 89 ... 6765, chauffe_eps = 20 000 + k),
a un delta fixe bien au-dessus des seuils. Fichiers de lancer_tour59_hasard.sh :
    /d/tmp/rdtrl_t59_<prefixe>_k<k>_<eps>_<offset>.txt
Sort, par eps : temps min / mediane / max, sd/moyenne (cv), et la part des phases
qui ne cassent pas dans le budget. Un passage deterministe donne cv ~ 0 ; un
echappement stochastique a taux constant donne cv ~ 1.
"""

import glob
import re
import sys
from collections import defaultdict

import numpy as np

if __name__ == "__main__":
    prefixe = sys.argv[1] if len(sys.argv) > 1 else "fr"
    par_eps = defaultdict(list)
    for f in glob.glob(f"D:/tmp/rdtrl_t59_{prefixe}_k*.txt"):
        m = re.search(r"_k(\d+)_([0-9.e+-]+)_([0-9.e+-]+)\.txt$", f)
        txt = open(f).read()
        if not m or "bascule" not in txt:
            continue
        eps, off = m.group(2), m.group(3)
        b = int(re.search(r"bascule=(-?\d+)", txt).group(1))
        par_eps[(float(eps), off)].append((int(m.group(1)), b))
    print("eps      offset   phases  cassent   min     mediane   max      moyenne   sd/moyenne   (temps en pas)")
    for (eps, off), L in sorted(par_eps.items()):
        t = np.array([b for _, b in L if b >= 0], dtype=float)
        n = len(L)
        if len(t) < 3:
            print(f"{eps:.0e}  {off:>7s}  {n:5d}  {len(t):6d}   (moins de 3 evenements)")
            continue
        print(f"{eps:.0e}  {off:>7s}  {n:5d}  {len(t):6d}  {t.min():7.0f} {np.median(t):8.0f} {t.max():7.0f}  {t.mean():8.0f}   {t.std(ddof=1) / t.mean():7.3f}")
