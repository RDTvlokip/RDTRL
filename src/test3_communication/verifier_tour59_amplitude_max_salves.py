"""Tour 59 (28/09/2026) : H59-16. A eps 1e-10 la bascule est-elle une
falaise "amplitude maximale des salves contre hauteur de barriere" ?

Sur les traces chauffees de verifier_tour59_delta_c_dynamique.py (eps
1e-10, chauffe 20 000 + 20 000, 40 000 pas) aux offsets a pli-1e-8,
+5e-9, +6e-9 (tiennent 40 000 pas), on calcule pour d3 = 1-s3 et
r4 = R[msg,b]'s receiver share la fraction x = (valeur - noeud) /
(jumeau - noeud) : maximum et quantile 99,9 % par fenetre de 10 000
pas. H59-16 predit max(x) ~ 1 (0,3 < max < 3) a +6e-9 ; le 27/09
(protocole court) donnait deja 3,9 a pli-1e-8 sans bascule.
Colonnes de la trace : pas, 1-s3, R[msg,b], r[msg,b], sqrt(v) x4.
"""

import glob
import re
import numpy as np

from verifier_tour59_branches_fermees import branche_et_jumeau, mesures, F_PLI

if __name__ == "__main__":
    fichiers = glob.glob("D:/tmp/rdtrl_tour59_trace_mur23_a3_b4_delta*_pas40000_eps1e-10_chauffe20000_20000.txt")
    for f in sorted(fichiers, key=lambda s: float(re.search(r"delta([0-9.e-]+)_pas", s).group(1))):
        d = float(re.search(r"delta([0-9.e-]+)_pas", f).group(1))
        T = np.loadtxt(f)
        if d > F_PLI:
            # au-dessus du pli il n'y a plus de noeud ni de jumeau : reference = le point
            # de pli lui-meme (les deux racines confondues, calculees a pli-1e-12).
            x0 = T[:, 1]
            rs = branche_et_jumeau(F_PLI - 1e-12)
            dF, _, rF = mesures(rs[0], F_PLI - 1e-12)
            print(f"pli{d - F_PLI:+.1e}  AU-DESSUS du pli, aucun point fixe. Reference = point de pli : d3={dF:.6e} r4={rF:.6f}")
            for nom, v, ref in (("d3", T[:, 1], dF), ("r4", T[:, 3], rF)):
                fen = [(np.median(v[i:i + 10000]) - ref, np.quantile(v[i:i + 10000], 0.001) - ref, np.quantile(v[i:i + 10000], 0.999) - ref)
                       for i in range(0, len(v), 10000)]
                print(f"    {nom} - pli : mediane par fenetre " + " ".join(f"{m:+.2e}" for m, _, _ in fen)
                      + f" | q0,1 % {min(q for _, q, _ in fen):+.2e} | q99,9 % {max(q for _, _, q in fen):+.2e}")
            continue
        rs = branche_et_jumeau(d)
        (dA, _, rA), (dJ, _, rJ) = mesures(rs[0], d), mesures(rs[1], d)
        xd, xr = (T[:, 1] - dA) / (dJ - dA), (T[:, 3] - rA) / (rJ - rA)
        print(f"pli{d - F_PLI:+.1e}  ecart d3 nœud-jumeau {dJ - dA:.2e}")
        for nom, x in (("d3", xd), ("r4", xr)):
            fen = [(x[i:i + 10000].max(), np.quantile(x[i:i + 10000], 0.999)) for i in range(0, len(x), 10000)]
            print(f"    {nom} : max par fenetre de 10 000 = " + " ".join(f"{m:+.2f}" for m, _ in fen)
                  + "   q99,9 % = " + " ".join(f"{q:+.2f}" for _, q in fen))
