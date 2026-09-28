"""Tour 59 (28/09/2026) : H59-12 (Delta(eps) = -c * skew(eps)) et H59-13
(sqrt(v_lent) = 5,1e-8, lu sur les logits de la collision).

skew(eps) = mean - median de d3 = 1-s3 sur les 30 000 derniers pas d'une
trace de 40 000 pas a pli-1e-6 (protocole de la grille : chauffe 20 000
+ chauffe_eps 20 000, meme etat chauffe). Delta(eps) vient de
verifier_tour59_ajustement_fantome.py (loi du fantome ajustee, si
>= 3 points basculent ; a trois points c'est une inversion, pas un test).
sqrt(v) : moyenne geometrique, sur les quatre logits e[3,10], e[4,10],
r[10,3], r[10,4], de sqrt(exp_avg_sq), temps-moyennee sur la meme fenetre.
"""

import glob
import re
import numpy as np

from verifier_tour59_ajustement_fantome import lire, ajuster
from verifier_tour59_branches_fermees import branche_et_jumeau, mesures

if __name__ == "__main__":
    runs = lire()
    delta_par_eps = {}
    for eps, lst in runs.items():
        casse = [(o, b) for o, b, _ in lst if b >= 0]
        if len(casse) >= 3:
            off = np.array([o for o, _ in casse]); t = np.array([b for _, b in casse], dtype=float)
            r2, t0, A, dc = ajuster(off, t)
            delta_par_eps[eps] = (dc, 1 / np.sqrt(A), len(casse))
    def cle(s):
        return float(re.search(r"_eps([0-9.e+-]+)_chauffe", s).group(1))
    for f in sorted(glob.glob("D:/tmp/rdtrl_tour59_trace_mur23_a3_b4_delta*_pas40000_eps*_chauffe20000_20000.txt"), key=cle):
        d = float(re.search(r"delta([0-9.e-]+)_pas", f).group(1))
        eps = re.search(r"_eps([0-9.e+-]+)_chauffe", f).group(1).replace("e-0", "e-")
        T = np.loadtxt(f)[-30000:]
        rs = branche_et_jumeau(d)
        dA = mesures(rs[0], d)[0]
        x = T[:, 1] - dA
        skew = x.mean() - np.median(x)
        sv = T[:, 4:8]
        cols = np.exp(np.log(sv).mean(axis=0))
        gm = np.exp(np.log(sv).mean())
        print(f"eps={eps:6s}  skew={skew:+.3e}  sd(d3)={x.std():.3e}  sqrt(v) moy. geom. 4 logits={gm:.3e} "
              f"(par logit {' '.join(f'{c:.2e}' for c in cols)})")
        if eps in delta_par_eps and skew != 0:
            dc, kap, n = delta_par_eps[eps]
            print(f"           Delta={dc:+.2e} (kappa {kap:.3f}, {n} points)   Delta/(-skew)={dc / -skew:+.4f}")
