"""Tour 59 : dans les traces pas a pas juste sous le pli (mur 23), quelle
fraction de la distance branche stable -> jumeau instable les excursions
de d3 = 1-s3 atteignent-elles, et ou (quel pas) ? Et y a-t-il
depassement de la branche stable (approche par en dessous) ?"""

import glob
import re
import numpy as np
from verifier_tour59_branches_fermees import branche_et_jumeau, mesures

for f in sorted(glob.glob("D:/tmp/rdtrl_tour59_trace_mur23_a3_b4_delta*_pas40000.txt")):
    d = float(re.search(r"delta([0-9.e-]+)_pas", f).group(1))
    T = np.loadtxt(f)
    t, d3, Rb, r4 = T[:, 0], T[:, 1], T[:, 2], T[:, 3]
    rs = branche_et_jumeau(d)
    (dA, RA, rA), (dJ, RJ, rJ) = mesures(rs[0], d), mesures(rs[1], d)
    frac = (d3 - dA) / (dJ - dA)
    fr4 = (r4 - rA) / (rJ - rA)
    i1 = np.argmax(np.abs(d3 - dA) < 0.01 * (dJ - dA))  # premier pas a 1 % de la distance au jumeau
    print(f"delta={d:.8f}  ecart jumeau d3={dJ - dA:.3e}  frac max d3={frac.max():+.4f} au pas {int(t[frac.argmax()])}   "
          f"frac max r4={fr4.max():+.4f} au pas {int(t[fr4.argmax()])}   premier pas a 1 %={int(t[i1])}   "
          f"frac finale d3={frac[-1]:+.4f}")
    for a, b in ((0, 2000), (2000, 10000), (10000, 20000), (20000, 30000), (30000, 40000)):
        s = frac[a:b]
        print(f"      pas [{a},{b}) frac d3 min {s.min():+.4f} max {s.max():+.4f}")
