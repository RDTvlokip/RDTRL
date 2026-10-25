"""Tour 88 : le plancher de protocole est-il le meme a chaque lr ? Seule compte la part qui differe d'un lr a l'autre.

Il dit : un biais de protocole commun a tous les lr s'annule dans tout exposant ; la difference bas - haut s'appuie surtout sur lr 0,2 (poids +1,35 dans le segment bas, -2,54 dans le haut, soit +3,89) ; abaisser le niveau de lr 0,2 de 0,10 % fait disparaitre la marche (0,0198 -> 0,0163 et 0,0118 -> 0,0163) et fait tomber bas - haut de 0,006 a 0,002 ; il demande de varier la fenetre et le pas de balayage aux cinq lr et de rapporter le decalage de chaque exposant local, et de lire lr 0,225.

Modes :
  arithmetique : ses chiffres a partir des niveaux du tour 87, et l'influence de +0,1 % a chaque lr sur chaque exposant local et sur bas - haut (aucune donnee lue)
  variantes    : lectures directes (60 salves par depart, memes salves, dernier franchissement) a pas 1 et 6 et a fenetre 210 contre la reference (pas 3, fenetre 140) ; decalage par lr et ses consequences
  grille       : niveaux d'enregistrement entier (C' x moyenne de 1/rho) sur la grille 0,1 a 0,3 de pas 0,025, ecart a la corde 0,15-0,25
  decalage_rho : niveaux d'enregistrement avec rho pris 1 pas ou 30 pas avant le groupe
"""

import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, ".")
from verifier_tour84_coupures import droite  # noqa: E402

D = "D:/tmp"
LRS5 = [0.1, 0.15, 0.2, 0.25, 0.3]
MOTIF = re.compile(r"salve (\d+) : calme avant (\d+) pas ; duree (\d+) pas ; lr p_z ([0-9.e+]+) ; sqrt\(v_e3\) ([0-9.e+\-]+) ; avance ([0-9.\-]+) pas")
# niveaux d'enregistrement entier du tour 87 (moyenne de dix departs) et leur incertitude (entre departs + bloc, quadrature, en %)
NIVEAUX87 = np.array([3.24924e6, 3.27187e6, 3.29052e6, 3.29919e6, 3.30658e6])
SE87 = np.array([math.hypot(0.003, 0.005), math.hypot(0.002, 0.010), math.hypot(0.006, 0.018), math.hypot(0.010, 0.028), math.hypot(0.011, 0.033)]) / 100


def exposants_locaux(lrs, niv):
    return [math.log(niv[i + 1] / niv[i]) / math.log(lrs[i + 1] / lrs[i]) for i in range(len(lrs) - 1)]


def segments(lrs, niv, se):
    """(exposant bas 0,1-0,2, exposant haut 0,2-0,3, bas - haut) ; se : incertitude relative par point (poids 1/se^2) ou None pour non pondere."""
    x = np.log(lrs); y = np.log(niv)
    s = np.ones(len(lrs)) if se is None else se
    ib = [0, 1, 2]; ih = [2, 3, 4]
    eb = droite(x[ib], y[ib], s[ib])[0]; eh = droite(x[ih], y[ih], s[ih])[0]
    return eb, eh, eb - eh


def arithmetique():
    lrs = np.array(LRS5)
    x = np.log(lrs)
    for nom, ix in (("bas (0,1 ; 0,15 ; 0,2)", [0, 1, 2]), ("haut (0,2 ; 0,25 ; 0,3)", [2, 3, 4])):
        xs = x[ix]; w = (xs - xs.mean()) / ((xs - xs.mean()) ** 2).sum()
        print(f"poids non pondere des trois points dans la pente {nom} : {np.round(w, 3)}")
    print(f"  lr 0,2 dans bas - haut : {1.348:+.2f} - ({-2.54:+.2f}) = {1.348 + 2.54:+.2f} (lui : +3,89)")
    e = exposants_locaux(lrs, NIVEAUX87)
    print("exposants locaux a partir des niveaux du tour 87 : " + " ".join(f"{v:+.4f}" for v in e))
    niv = NIVEAUX87.copy(); niv[2] *= math.exp(-0.001)
    e2 = exposants_locaux(lrs, niv)
    print("apres avoir abaisse le niveau de lr 0,2 de 0,10 % : " + " ".join(f"{v:+.4f}" for v in e2))
    for nom, se in (("non pondere", None), ("pondere par l'incertitude du tour 87 (bruit + entre departs)", SE87), ("pondere avec plancher 0,1 %", np.hypot(SE87, 0.001))):
        eb, eh, d0 = segments(lrs, NIVEAUX87, se)
        eb2, eh2, d1 = segments(lrs, niv, se)
        print(f"  bas - haut, {nom} : {d0:+.4f} (bas {eb:+.4f}, haut {eh:+.4f}) ; apres -0,10 % a lr 0,2 : {d1:+.4f}")
    print("influence de +0,10 % du niveau de chaque lr sur chaque exposant local et sur bas - haut :")
    for i, lr in enumerate(LRS5):
        n3 = NIVEAUX87.copy(); n3[i] *= math.exp(0.001)
        el = exposants_locaux(lrs, n3)
        ligne = f"   lr {lr}: locaux " + " ".join(f"{a - b:+.4f}" for a, b in zip(el, e))
        for nom, se in (("non pondere", None), ("pondere", SE87)):
            ligne += f" ; bas - haut ({nom}) {segments(lrs, n3, se)[2] - segments(lrs, NIVEAUX87, se)[2]:+.4f}"
        print(ligne)
    # l'ecart de lr 0,2 a la corde 0,15-0,25
    ec = math.log(NIVEAUX87[1]) + math.log(NIVEAUX87[3] / NIVEAUX87[1]) * math.log(0.2 / 0.15) / math.log(0.25 / 0.15)
    print(f"niveau de lr 0,2 au-dessus de la corde 0,15-0,25 (log-log) : {100 * (math.log(NIVEAUX87[2]) - ec):+.3f} %")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "arithmetique":
        arithmetique()
