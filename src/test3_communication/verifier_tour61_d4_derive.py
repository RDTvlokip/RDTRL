"""Tour 61 (29/09/2026) : le logit e4 - o4 de la ligne 4 (d4 = 1 - s4) est-il gele ou
derive-t-il lentement vers son noeud (3,1e-17) sous le plancher d'Adam ?

Raisonnement : au noeud X4 = z4 ; loin de lui, ge4 = s4 d4 B (X4 - z4)/N ~ d4 (X4 - z4) B/N.
Si |g| << eps (1e-10), Adam agit comme SGD de pas lr/eps : dz4/dt = (lr/eps) g ~ K d4, et
d4 = C e^-z4, donc dz4/dt = K' e^-z4, soit e^z4 = K' t + c, d4 ~ 1/(t + t0) : une derive
algebrique, non un gel. Prediction (mise 55 %) : d4 mediane par fenetre suit t^-a avec a
dans [0,8 ; 1,2] sur 3 000 000 pas apres l'etat chaud, a delta = pli-1e-9.
Reduction pure Python (reduction_deux_lignes_tour59), etat chaud 40 000 pas du reseau complet.
"""

import sys
import numpy as np
from mpmath import mpf

sys.path.insert(0, ".")
from reduction_deux_lignes_tour59 import charger, simuler
from verifier_tour60_exposant_reduction import PLI

CHAUD = "D:/tmp/rdtrl_tour59_chaud_mur23_3_4_eps1e-10_20000_20000.pt"

if __name__ == "__main__":
    s0 = charger(CHAUD)
    delta = float(PLI - mpf("1e-9"))
    fenetre, nfen = 200_000, 15
    fin = s0
    tt, med = [], []
    print(f"delta = pli-1e-9 ; fenetres de {fenetre} pas apres l'etat chaud (t compte depuis les 40 000 pas du reseau complet)")
    print("t central (pas)   d4 mediane      d4 min / max")
    for i in range(nfen):
        out, b, fin = simuler(fin, delta, 1e-10, fenetre, trace=True, seuil_stop=False)
        d4 = out[:, 4]
        t = 40000 + (i + 0.5) * fenetre
        tt.append(t); med.append(np.median(d4))
        print(f"{t:12.0f}     {np.median(d4):.4e}     {d4.min():.3e} / {d4.max():.3e}", flush=True)
    tt, med = np.array(tt), np.array(med)
    a, b = np.polyfit(np.log(tt[2:]), np.log(med[2:]), 1)
    print(f"\najustement log-log (fenetres 3 a {nfen}) : d4 ~ t^{a:.3f}   (prediction : exposant -1 +- 0,2)")
    # forme 1/(t + t0) : 1/d4 lineaire en t
    p = np.polyfit(tt, 1.0 / med, 1)
    res = np.max(np.abs((p[0] * tt + p[1]) * med - 1.0))
    print(f"1/d4 lineaire en t : 1/d4 = {p[0]:.4e} t + {p[1]:.4e} ; t0 = {p[1] / p[0]:.3e} ; ecart relatif max de l'ajustement {res:.3f}")
