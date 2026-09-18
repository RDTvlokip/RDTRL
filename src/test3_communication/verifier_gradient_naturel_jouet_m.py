"""Gradient naturel pour le jouet a M categories de fond (CARNET.md
§7.65, 18/09/2026) : compense l'aplatissement structurel du sigmoide/
softmax (le gradient brut s'annule comme s3(1-s3) cote emetteur,
r3*r4 cote recepteur -- cf. verifier_sgd_pur_jouet_m.py) en divisant
chaque mise a jour par ce meme facteur.

HISTORIQUE DE CE FIL (resolu, cloture) : un premier essai (emetteur
seul en gradient naturel, recepteur en SGD brut) a semble montrer une
divergence qualitative avec Adam (s3 grimpe vers 1 au lieu de se
stabiliser a 0,9964) -- fausse alerte, tracee a une comparaison a la
mauvaise variable (R algebrique au lieu de R mesure), puis a un choix
de coordonnee invalide (x au lieu de z3, espace logit -- la relation
dz3/dt=lr*(poids3*r3-(beta/N)*z3) est EXACTE, verifiee a ratio=1,0000).

Cloture finale : une relation exacte symetrique existe cote RECEPTEUR
(u=q3-q4, r3=sigmoid(u)) :
    u* = (1/beta) * [(1-delta)*s3 - (1+delta)*s4]
    r4* = 1 - sigmoid(u*)
verifiee a 2,4e-6 pres contre le plateau Adam observe. Le systeme
COUPLE (emetteur ET recepteur en gradient naturel) converge vers
EXACTEMENT le meme point fixe qu'Adam (s3 et r4 a 7 chiffres
significatifs, ecart 1,845e-7 a la prediction) -- a condition d'un lr
suffisant (0,02 insuffisant, lr=5,0 converge en 40000 pas). Adam
n'est qu'un accelerateur, pas un mecanisme differencie -- confirme
definitivement.
"""

import sys
sys.path.insert(0, '.')
import torch

from verifier_jouet_n_variable import construire_toy_m, objectif_toy_m, N

DELTA_C_M0 = 0.01867676
DELTA_C_M25 = 0.01855957


def trace_gradient_naturel(M, delta, lr, r_autres_init, pas, points, s3_init):
    poids3 = torch.tensor((1.0 - delta) / N, dtype=torch.float64)
    poids4 = torch.tensor((1.0 + delta) / N, dtype=torch.float64)
    p3, p4, q = construire_toy_m(M, delta, r_autres_init=r_autres_init, s3_init=s3_init)
    out = []
    pset = set(points)
    for i in range(pas):
        j = objectif_toy_m(p3, p4, q, poids3, poids4)
        for p in (p3, p4, q):
            if p.grad is not None:
                p.grad = None
        (-j).backward()
        with torch.no_grad():
            s3 = torch.sigmoid(p3).item()
            s4 = torch.sigmoid(p4).item()
            denom3 = max(s3 * (1 - s3), 1e-300)
            denom4 = max(s4 * (1 - s4), 1e-300)
            p3 -= lr * p3.grad / denom3
            p4 -= lr * p4.grad / denom4
            q -= lr * q.grad  # recepteur : SGD normal, deja fluide sans blocage
        if (i + 1) in pset:
            with torch.no_grad():
                s3n = torch.sigmoid(p3).item()
                r_all = torch.softmax(q, dim=0)
                r4 = r_all[1].item()
                masse_autres = 1.0 - r_all[0].item() - r4
            out.append((i + 1, s3n, r4, masse_autres))
    return out


def trace_double_gradient_naturel(M, delta, lr, r_autres_init, pas, points, s3_init):
    """Gradient naturel des DEUX cotes (emetteur ET recepteur) -- la
    piece manquante du premier essai (recepteur en SGD brut). Cote
    recepteur (M=0, softmax a 2 voies) : diviser par r3*r4, le meme
    facteur d'aplatissement que r3(1-r3) puisque r4=1-r3."""
    poids3 = torch.tensor((1.0 - delta) / N, dtype=torch.float64)
    poids4 = torch.tensor((1.0 + delta) / N, dtype=torch.float64)
    p3, p4, q = construire_toy_m(M, delta, r_autres_init=r_autres_init, s3_init=s3_init)
    out = []
    pset = set(points)
    for i in range(pas):
        j = objectif_toy_m(p3, p4, q, poids3, poids4)
        for p in (p3, p4, q):
            if p.grad is not None:
                p.grad = None
        (-j).backward()
        with torch.no_grad():
            s3 = torch.sigmoid(p3).item()
            s4 = torch.sigmoid(p4).item()
            r_all = torch.softmax(q, dim=0)
            r3v, r4v = r_all[0].item(), r_all[1].item()
            denom3 = max(s3 * (1 - s3), 1e-300)
            denom4 = max(s4 * (1 - s4), 1e-300)
            denom_r = max(r3v * r4v, 1e-300)
            p3 -= lr * p3.grad / denom3
            p4 -= lr * p4.grad / denom4
            q -= lr * q.grad / denom_r
        if (i + 1) in pset:
            with torch.no_grad():
                s3n = torch.sigmoid(p3).item()
                r4 = torch.softmax(q, dim=0)[1].item()
            out.append((i + 1, s3n, r4))
    return out


def trace_adam_long(M, delta, lr, r_autres_init, pas, points):
    poids3 = torch.tensor((1.0 - delta) / N, dtype=torch.float64)
    poids4 = torch.tensor((1.0 + delta) / N, dtype=torch.float64)
    p3, p4, q = construire_toy_m(M, delta, r_autres_init=r_autres_init)
    opt = torch.optim.Adam([p3, p4, q], lr=lr, eps=1e-10)
    out = []
    pset = set(points)
    for i in range(pas):
        j = objectif_toy_m(p3, p4, q, poids3, poids4)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if (i + 1) in pset:
            with torch.no_grad():
                s3 = torch.sigmoid(p3).item()
                r_all = torch.softmax(q, dim=0)
                r4 = r_all[1].item()
            out.append((i + 1, s3, r4))
    return out


if __name__ == "__main__":
    points = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000]

    print("=== gradient naturel (emetteur), lr=0,02, s3_init=0,999, M=0 ===")
    for t, s3, r4, ma in trace_gradient_naturel(0, DELTA_C_M0, 0.02, 0.01, 40000, points, 0.999):
        print(f"  t={t:6d}  s3={s3:.8f}  r4={r4:.8f}  masse_autres={ma:.3e}")

    print()
    print("=== gradient naturel, lr=0,02, s3_init=0,999, M=25 ===")
    for t, s3, r4, ma in trace_gradient_naturel(25, DELTA_C_M25, 0.02, 0.01, 40000, points, 0.999):
        print(f"  t={t:6d}  s3={s3:.8f}  r4={r4:.8f}  masse_autres={ma:.3e}")

    print()
    print("=== Adam etendu a 400000 pas (verification que le plateau tient) ===")
    points_long = [40000, 60000, 100000, 150000, 200000, 300000, 400000]
    for t, s3, r4 in trace_adam_long(0, DELTA_C_M0, 0.2, 0.01, 400000, points_long):
        print(f"  t={t:7d}  s3={s3:.10f}  r4={r4:.10f}")

    print()
    print("=== point fixe recepteur, forme fermee (verifie contre Adam) ===")
    s3_adam, s4_adam = 0.9964323591, 0.9999999999992756
    beta = 0.02
    u_star = (1 / beta) * ((1 - DELTA_C_M0) * s3_adam - (1 + DELTA_C_M0) * s4_adam)
    import math
    r3_star = 1 / (1 + math.exp(-u_star))
    r4_star = 1 - r3_star
    print(f"  u*={u_star:.10f}  r3*={r3_star:.10f}  r4*={r4_star:.10f}")
    print(f"  (plateau Adam observe : r4=0,8852130348, ecart={abs(r4_star-0.8852130348):.3e})")

    print()
    print("=== systeme COUPLE (emetteur+recepteur en gradient naturel), lr=5,0 ===")
    for t, s3, r4 in trace_double_gradient_naturel(0, DELTA_C_M0, 5.0, 0.01, 40000, points, 0.999):
        print(f"  t={t:6d}  s3={s3:.8f}  r4={r4:.8f}  ecart_r4*={abs(r4-r4_star):.3e}")
