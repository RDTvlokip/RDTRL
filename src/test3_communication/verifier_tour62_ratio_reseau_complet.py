"""Tour 62 (07/10/2026) : le rapport delta'_med(r4)/delta'_med(d3) a ete mesure dans la
reduction a deux lignes (verifiee contre le reseau complet pour s, pas pour ce rapport).
Ici : le meme rapport depuis les traces du RESEAU COMPLET (E60-1/E60-3, eps 1e-10, mur 23,
6 phases k = 89 ... 987, 30 000 derniers pas de 40 000). Colonnes de la trace : pas, 1-s3,
R_b, r4 = r[msg, b]. Medianes exactes par coordonnee (d3 et r4 monotones en z3 et en u).
Prediction precommise : accord avec la reduction a 0,003 pres a pli-1e-9 (0,9415),
pli-3e-10 (0,9554) et pli-1e-10 (0,9612).
"""

import glob
import re
import sys
import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, ".")
from verifier_tour61_decomposition_d3 import PLI, d3_noeud, r4_noeud, pente_fn

mp.dps = 40
REDUCTION = {"-1e-09": 0.94150, "-3e-10": 0.95544, "-1e-10": 0.96123}

if __name__ == "__main__":
    groupes = {}
    for f in glob.glob("D:/tmp/rdtrl_tour59_trace_mur23_a3_b4_delta*_pas40000_eps1e-10_chauffe20000_*.txt"):
        m = re.search(r"delta([0-9.e-]+)_pas40000_eps1e-10_chauffe20000_(\d+)\.txt$", f)
        if not m or int(m.group(2)) == 20000:
            continue
        off = float(mpf(m.group(1)) - PLI)
        groupes.setdefault(round(off, 13), []).append(f)
    print("offset     phases   rapport reseau complet (moy +- ES)   reduction (24 phases)   ecart")
    for off in sorted(groupes, key=lambda o: -abs(o)):
        if abs(off) > 1.5e-9:
            continue
        dn, rn = float(d3_noeud(PLI + mpf(off))), float(r4_noeud(PLI + mpf(off)))
        sl3, sl4 = pente_fn(d3_noeud, PLI + mpf(off)), pente_fn(r4_noeud, PLI + mpf(off))
        rap = []
        for f in groupes[off]:
            T = np.loadtxt(f)[-30000:]
            rap.append(((np.median(T[:, 3]) - rn) / sl4) / ((np.median(T[:, 1]) - dn) / sl3))
        rap = np.array(rap)
        cle = min(REDUCTION, key=lambda k: abs(float(k) - off))
        ref = REDUCTION[cle] if abs(float(cle) - off) < 0.2 * abs(off) else None
        print(f"{off:+.0e}   {len(rap):3d}      {rap.mean():.5f} +- {rap.std(ddof=1) / np.sqrt(len(rap)):.5f}"
              + (f"           {ref:.5f}              {rap.mean() - ref:+.5f}" if ref else ""))
