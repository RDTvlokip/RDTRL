"""Tour 59 : decomposer les fluctuations (X3, r4) sous le pli dans la base
{direction lente = stable -> jumeau, direction rapide = axe principal des
oscillations}. Si les salves ne franchissent pas malgre X3 et r4 au-dela
du jumeau, c'est que leur composante LENTE reste petite (H59-6).
Lit la trace de verifier_tour59_jensen_concurrents.py.

RESULTAT NON CONCLUANT (27/09) : les deux directions sont presque
paralleles (pentes dr4/dX3 -0,020 et +0,002, toutes deux >99,9 % selon
X3), la base est mal conditionnee et la "composante lente +-18" n'est
que du bruit amplifie. Remplace par verifier_tour59_biais_moyen_salves.py
(mediane et quantiles de d3 contre le noeud ferme)."""

import sys
import numpy as np
from verifier_tour59_branches_fermees import branche_et_jumeau, F_PLI

off = float(sys.argv[1]) if len(sys.argv) > 1 else -1e-8
d = F_PLI + off
T = np.loadtxt(f"D:/tmp/rdtrl_tour59_jensen_mur23_off{off}.txt")
rs = branche_et_jumeau(d)
r3 = lambda u: 1 / (1 + np.exp(-u))
XA, XJ = (1 - d) * r3(rs[0]) / 0.02, (1 - d) * r3(rs[1]) / 0.02
rA, rJ = 1 - r3(rs[0]), 1 - r3(rs[1])
P = np.column_stack([T[:, 3] - XA, T[:, 4] - rA])
lent = np.array([XJ - XA, rJ - rA])
C = np.cov(P.T)
w, V = np.linalg.eigh(C)
rapide = V[:, -1]
print(f"direction lente (dX3, dr4) = {lent / np.linalg.norm(lent)}   pente dr4/dX3 = {lent[1] / lent[0]:.4f}")
print(f"axe principal des oscillations = {rapide}   pente dr4/dX3 = {rapide[1] / rapide[0]:.4f}   "
      f"variance expliquee {w[-1] / w.sum():.6f}")
M = np.column_stack([lent, rapide])
coef = np.linalg.solve(M, P.T)
a = coef[0]
print(f"composante lente (1 = jumeau) : min {a.min():+.4f} max {a.max():+.4f} moyenne {a.mean():+.4f} ecart-type {a.std():.4f}")
print(f"composante rapide (unites de l'axe) : ecart-type {coef[1].std():.3e}")
# periode des oscillations
x = T[:, 3] - T[:, 3].mean()
f = np.abs(np.fft.rfft(x)); k = f[1:].argmax() + 1
print(f"periode dominante de X3 : {len(x) / k:.1f} pas")
