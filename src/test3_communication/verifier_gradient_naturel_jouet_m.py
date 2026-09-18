"""Gradient naturel pour l'emetteur du jouet a M categories de fond
(CARNET.md §7.65, 18/09/2026) : compense l'aplatissement structurel du
sigmoide (le gradient brut s'annule comme s3(1-s3) pres de s3=1, cf.
verifier_sgd_pur_jouet_m.py) en divisant la mise a jour de p3/p4 par
ce meme facteur.

Resultat obtenu : ca marche (l'emetteur bouge enfin sans Adam), mais
donne un comportement QUALITATIVEMENT different d'Adam -- s3 grimpe
en continu vers 1 au lieu de se stabiliser a 0,9964. Verification
croisee (Adam etendu a 400000 pas) montre que le plateau Adam a
0,9964 est un vrai point fixe robuste, pas un artefact de blocage
transitoire -- donc la divergence entre gradient naturel et Adam
n'est pas resolue par "il faut juste attendre plus longtemps" d'un
cote ou de l'autre. Reste une vraie question ouverte : la reduction
quasi-statique (x,R) (verifier_ode_jouet_m.py) elle-meme s'est
revelee incoherente avec la simulation directe hors du voisinage du
pli (F(x) ne s'annule jamais vers x=0, contrairement a ce que la
simulation montre) -- demande de refaire la derivation de R_br(x)
sans supposer le recepteur a l'equilibre instantane, pas tente ici.
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
