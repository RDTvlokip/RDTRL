"""Tour 81 : statistiques entre departs de lr p_z, d_s/niveau et du rapport, x = 1e-6, niveau exact, lus dans les journaux /d/tmp/t81_lr<lr>_k<k>.log (lancer_tour81.sh, lancer_tour81b.sh).
Donne par lr : n departs, moyenne, ecart-type, minimum, maximum de lr p_z ; exposants locaux de la moyenne ; ecart de la moyenne a 0,15 a la droite log (lr 0,1 - lr 0,2) ; pour chaque depart a lr 0,15, l'ecart a 2,3376e6.
Usage : python verifier_tour81_stats.py [dossier]
"""

import glob
import math
import re
import sys

import numpy as np

dossier = sys.argv[1] if len(sys.argv) > 1 else "D:/tmp"
LRS = ["0.1", "0.125", "0.15", "0.175", "0.2", "0.3"]
val = {lr: {} for lr in LRS}
for lr in LRS:
    for f in sorted(glob.glob(f"{dossier}/t81_lr{lr}_k*.log")):
        k = int(re.search(r"_k(\d+)\.log", f).group(1))
        t = open(f, encoding="utf-8", errors="replace").read()
        m = re.search(r"lr p_z = ([0-9.e+\-]+)", t)
        d = re.search(r"d_s/niveau = ([0-9.]+)", t)
        r = re.search(r"rapport ([0-9.]+)", t)
        n = re.search(r"\((\d) salves\)", t)
        if m and d and r:
            val[lr][k] = (float(m.group(1)), float(d.group(1)), float(r.group(1)), int(n.group(1)) if n else 0)

print("lr     n  lr p_z moyen   sd (%)  min        max        d_s/niveau moyen  rapport moyen   departs (lr p_z)")
moy = {}
for lr in LRS:
    v = val[lr]
    if not v:
        continue
    pz = np.array([a[0] for a in v.values()])
    moy[float(lr)] = pz.mean()
    print(f"{lr:6s} {len(pz)}  {pz.mean():.4e}   {100 * pz.std(ddof=1) / pz.mean() if len(pz) > 1 else 0:.2f}   {pz.min():.4e} {pz.max():.4e}   {np.mean([a[1] for a in v.values()]):.5f}   {np.mean([a[2] for a in v.values()]):.4f}   "
          + " ".join(f"k{k}:{a[0] / 1e6:.4f}" for k, a in sorted(v.items())))
ls = sorted(moy)
print("exposants locaux de la moyenne (d ln lr p_z / d ln lr) :")
for a, b in zip(ls[:-1], ls[1:]):
    print(f"   {a} -> {b} : {math.log(moy[b] / moy[a]) / math.log(b / a):.4f}")
if 0.1 in moy and 0.2 in moy and 0.15 in moy:
    ligne = moy[0.1] * (moy[0.2] / moy[0.1]) ** (math.log(1.5) / math.log(2))
    print(f"droite log (moyennes lr 0,1 - 0,2) a lr 0,15 : {ligne:.4e} ; moyenne a 0,15 : {moy[0.15]:.4e} ({100 * (moy[0.15] / ligne - 1):+.2f} %)")
if val["0.15"]:
    print("departs a lr 0,15 contre 2,3376e6 :", " ".join(f"k{k}:{100 * (a[0] / 2.3376e6 - 1):+.2f}%" for k, a in sorted(val["0.15"].items())))
