"""RDTRL — son recensement egalite/mur, reproduit sur mon propre code torch.

Il classe chaque collision par le partage de masse du recepteur sur le
message conteste : egalite si les deux membres sont dans (0,1 ; 0,9),
mur si l'un depasse 0,99 et l'autre est sous 0,01. Sur 30 graines, il trouve
42 egalites contre 8 murs (sur 50 collisions). Reproduit ici independamment,
generateur different, code different (torch plutot que sa reimplementation
numpy).
"""

import numpy as np
import torch

from grammaire3 import N
from representable_atteignable_stable import EmetteurTabulaire, Recepteur, monter, lire_code

BETA = 0.02
PAS = 20000
GRAINES = 30


def classer(e, r, code):
    with torch.no_grad():
        rr = r.loi()
    uniques, comptes = np.unique(code, return_counts=True)
    doublons = uniques[comptes > 1]
    egalites = murs = autres = 0
    for m in doublons:
        refs = np.where(code == m)[0]
        masses = sorted(float(rr[m, ref]) for ref in refs)
        if len(refs) != 2:
            autres += 1
            continue
        a, b = masses
        if 0.1 < a < 0.9 and 0.1 < b < 0.9:
            egalites += 1
        elif a < 0.01 and b > 0.99:
            murs += 1
        else:
            autres += 1
    return egalites, murs, autres, len(doublons)


if __name__ == "__main__":
    print(f"=== RECENSEMENT EGALITE/MUR, {GRAINES} graines, beta={BETA}, {PAS} pas ===")
    g = np.random.default_rng(31415)
    tot_e = tot_m = tot_a = tot_coll = 0
    for k in range(GRAINES):
        e, r = EmetteurTabulaire(g), Recepteur(g)
        monter(e, r, BETA, PAS, lr=0.05)
        code = lire_code(e)
        eg, mu, au, nc = classer(e, r, code)
        tot_e += eg; tot_m += mu; tot_a += au; tot_coll += nc
    print(f"\n  collisions totales : {tot_coll}")
    print(f"  egalites (0,1-0,9 des deux cotes) : {tot_e}")
    print(f"  murs (un >0,99, un <0,01)          : {tot_m}")
    print(f"  autres                              : {tot_a}")
