"""Tour 59 : sous le pli, l'etat oscille (periode ~444 pas, la salve du
bord de stabilite du tour 58) autour de la branche stable fermee. Le
biais MOYEN (pas 20 000 a 40 000) de d3 = 1-s3 et de r4 par rapport a la
branche stable est-il constant en valeur absolue (propriete de la salve,
independante de delta) ou proportionnel a l'ecart au jumeau ?
Lit les traces pas a pas de verifier_tour59_delta_c_dynamique.py (option trace)."""

import glob
import re
import numpy as np
from verifier_tour59_branches_fermees import branche_et_jumeau, mesures, F_PLI

if __name__ == "__main__":
    for f in sorted(glob.glob("D:/tmp/rdtrl_tour59_trace_mur23_a3_b4_delta*_pas40000.txt")):
        d = float(re.search(r"delta([0-9.e-]+)_pas", f).group(1))
        T = np.loadtxt(f)[20000:]
        rs = branche_et_jumeau(d)
        (dA, _, rA), (dJ, _, rJ) = mesures(rs[0], d), mesures(rs[1], d)
        b3, b4 = T[:, 1].mean() - dA, T[:, 3].mean() - rA
        print(f"pli{d - F_PLI:+.1e}  ecart jumeau d3 {dJ - dA:.3e}  biais moyen d3 {b3:+.3e} (sd {T[:, 1].std():.3e})   "
              f"ecart jumeau r4 {rJ - rA:.3e}  biais moyen r4 {b4:+.3e} (sd {T[:, 3].std():.3e})")
        # relaxation encore en cours ? biais par fenetre de 4440 pas (10 periodes de salve)
        fen = [T[i:i + 4440, 1].mean() - dA for i in range(0, len(T) - 4439, 4440)]
        print("      biais d3 par fenetre de 10 salves : " + "  ".join(f"{b:+.3e}" for b in fen))
        # salves asymetriques : l'etat de REPOS entre salves (mediane, quantiles) est-il sur le noeud ?
        q = np.quantile(T[:, 1] - dA, [0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"      d3 - noeud : quantiles 5/25/50/75/95 % = " + "  ".join(f"{x:+.3e}" for x in q)
              + f"   (en fraction de l'ecart au jumeau : mediane {q[2] / (dJ - dA):+.4f})")
