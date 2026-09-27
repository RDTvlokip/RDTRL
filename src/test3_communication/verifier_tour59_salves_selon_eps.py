"""Tour 59 : les salves du bord de stabilite existent-elles encore apres
chauffe (20 000 pas a pli-1e-6, eps 1e-10) puis bascule vers eps = 1e-8,
1e-7, 1e-6 ? Ecart-type et quantiles de d3 - noeud ferme sur les 10 000
derniers pas des traces de verifier_tour59_delta_c_dynamique.py
(options trace eps=... chauffe=20000, delta = pli - 1e-6)."""

import glob
import re
import numpy as np
from verifier_tour59_branches_fermees import branche_et_jumeau, mesures

if __name__ == "__main__":
    for f in sorted(glob.glob("D:/tmp/rdtrl_tour59_trace_mur23_a3_b4_delta*_pas20000_eps*_chauffe20000.txt")):
        d = float(re.search(r"delta([0-9.e-]+)_pas", f).group(1))
        eps = re.search(r"_eps([0-9.e+-]+)_chauffe", f).group(1)
        T = np.loadtxt(f)[-10000:]
        rs = branche_et_jumeau(d)
        (dA, _, _), (dJ, _, _) = mesures(rs[0], d), mesures(rs[1], d)
        x = T[:, 1] - dA
        q = np.quantile(x, [0.01, 0.5, 0.99])
        print(f"eps={eps:6s}  sd(d3)={x.std():.3e}  moyenne-mediane={x.mean() - q[1]:+.3e}  "
              f"q1/50/99 % = {q[0]:+.3e} {q[1]:+.3e} {q[2]:+.3e}   "
              f"max vers jumeau = {x.max() / (dJ - dA):+.4f} de l'ecart")
