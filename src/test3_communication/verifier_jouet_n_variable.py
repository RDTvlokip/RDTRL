"""Piste 3 de ETAT.md, la derniere de la liste : isoler proprement la
piste "les 25 autres lignes" pour la derive de k(R) -- pas avec le jouet
a 2 categories de verifier_jouet_2_referents.py (qui change N=27 lui-meme
autant que le nombre de "lignes participantes", donc pas une ablation
propre), mais avec un jouet a M CATEGORIES DE FOND VARIABLES sur le
recepteur, en gardant N=27 et beta=0,02 fixes dans la formule partout.

Modele : emetteur inchange (s3, s4 sigmoides independantes, forme deja
validee de H7). Recepteur etendu : au lieu d'un seul logit q (r4=sigmoid(q),
r3=1-r4, comme dans le jouet a 2 referents), un softmax a (2+M) categories
-- r3, r4, et M logits de fond ENTRAINES (pas fixes), sans aucune
recompense directe, seulement la pression d'entropie. Si M=0, on retrouve
exactement le jouet a 2 referents (verifier_jouet_2_referents.py) --
verifie ci-dessous comme test de non-regression avant de faire varier M.

Deux mesures a M croissant (0, 1, 3, 8, 25 -- 25 etant l'analogue direct
du systeme complet a 27 referents) :
  1. delta_c(M) -- interpole-t-il du delta_c du jouet pur (M=0, ~0,0187)
     vers celui du systeme complet (M=25, cense s'approcher de 0,0134372) ?
  2. l'ecart des points de bascule (s3 critique) entre R_init=0,50/0,60/0,75
     a delta fixe -- retrecit-il avec M croissant, signe que la derive de
     k(R) vient bien de la competition des 25 autres lignes ?
"""

import sys
sys.path.insert(0, '.')
import math
import torch

BETA = 0.02
N = 27


def construire_toy_m(M, delta, s3_init=0.999999999666, s4_init=0.999999999997,
                      r_tie=0.5, r_autres_init=1e-6):
    p3 = torch.tensor([math.log(s3_init / (1 - s3_init))], dtype=torch.float64, requires_grad=True)
    p4 = torch.tensor([math.log(s4_init / (1 - s4_init))], dtype=torch.float64, requires_grad=True)
    # logits du recepteur : [q3, q4, q_autre_1, ..., q_autre_M]
    if M > 0:
        part_autres = r_autres_init
        part_34 = 1.0 - M * part_autres
    else:
        part_34 = 1.0
    r3_init = part_34 * r_tie
    r4_init = part_34 * (1.0 - r_tie)
    logits = [math.log(max(r3_init, 1e-300)), math.log(max(r4_init, 1e-300))]
    logits += [math.log(r_autres_init)] * M
    q = torch.tensor(logits, dtype=torch.float64, requires_grad=True)
    return p3, p4, q


def objectif_toy_m(p3, p4, q, poids3, poids4):
    s3 = torch.sigmoid(p3)
    s4 = torch.sigmoid(p4)
    r_all = torch.softmax(q, dim=0)
    r3, r4 = r_all[0], r_all[1]
    recompense = poids3 * s3 * r3 + poids4 * s4 * r4
    ent = lambda p: -(p * torch.log(p.clamp_min(1e-300)) + (1 - p) * torch.log((1 - p).clamp_min(1e-300)))
    entropie_s = ent(s3) + ent(s4)
    entropie_r = -(r_all * torch.log(r_all.clamp_min(1e-300))).sum()
    return (recompense + (BETA / N) * (entropie_s + entropie_r)).sum()


def entrainer_toy_m(M, delta, pas=40000, lr=0.05, adam_eps=1e-10,
                     s3_init=0.999999999666, r_tie=0.5):
    poids3 = torch.tensor((1.0 - delta) / N, dtype=torch.float64)
    poids4 = torch.tensor((1.0 + delta) / N, dtype=torch.float64)
    p3, p4, q = construire_toy_m(M, delta, s3_init=s3_init, r_tie=r_tie)
    opt = torch.optim.Adam([p3, p4, q], lr=lr, eps=adam_eps)
    for _ in range(pas):
        j = objectif_toy_m(p3, p4, q, poids3, poids4)
        opt.zero_grad()
        (-j).backward()
        opt.step()
    with torch.no_grad():
        s3 = torch.sigmoid(p3).item()
        s4 = torch.sigmoid(p4).item()
        r_all = torch.softmax(q, dim=0)
        r3, r4 = r_all[0].item(), r_all[1].item()
        masse_autres = 1.0 - r3 - r4
        R4 = s4 * r4
    return R4, s3, s4, r3, r4, masse_autres


def bissecter_delta_c(M, lo=0.010, hi=0.020, tol=1e-4, pas=20000):
    def sature(delta):
        _, s3, *_ = entrainer_toy_m(M, delta, pas=pas)
        return s3 < 0.5

    s_lo, s_hi = sature(lo), sature(hi)
    assert s_lo != s_hi, f"M={M}: lo/hi meme issue ({s_lo}/{s_hi}), elargir le bracket"
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if sature(mid) == s_hi:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def bissecter_flip_s3(M, delta, r_tie_init, lo=0.90, hi=0.999999, tol=1e-4, pas=20000):
    def gradue(s3_init):
        _, s3f, *_ = entrainer_toy_m(M, delta, pas=pas, s3_init=s3_init, r_tie=r_tie_init)
        return s3f > 0.5

    g_lo, g_hi = gradue(lo), gradue(hi)
    assert g_lo != g_hi, f"M={M} r_tie={r_tie_init}: lo/hi meme issue"
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if gradue(mid) == g_lo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    print("=== test de non-regression : M=0 doit reproduire verifier_jouet_2_referents.py ===")
    for delta in (0.0, 0.013, 0.0187, 0.02):
        R4, s3, s4, r3, r4, masse_autres = entrainer_toy_m(0, delta)
        print(f"  M=0  delta={delta:<7}  R4={R4:.6f}  s3={s3:.6f}  s4={s4:.6f}  "
              f"r3={r3:.6f}  r4={r4:.6f}  masse_autres={masse_autres:.2e}")

    print("\n=== 1) delta_c(M) : interpole-t-il vers 0,0134372 (systeme complet) ? ===")
    Ms = (0, 3, 8, 25)
    deltas_c = {}
    for M in Ms:
        dc = bissecter_delta_c(M, tol=3e-4, pas=15000)
        deltas_c[M] = dc
        print(f"  M={M:<3}  delta_c={dc:.6f}   (jouet pur M=0: ~0,0187 ; systeme complet (M=25 vise): 0,013437)")

    print("\n=== 2) ecart des points de bascule entre R_init=0,50/0,75, a M croissant ===")
    print("    (delta fixe a 0,95*delta_c(M) pour rester comparable relativement au pli de CHAQUE M)")
    for M in Ms:
        dc = deltas_c[M]
        delta_test = 0.95 * dc
        flips = {}
        for r_tie in (0.50, 0.75):
            flip = bissecter_flip_s3(M, delta_test, r_tie, tol=3e-4, pas=15000)
            flips[r_tie] = flip
        spread = max(flips.values()) - min(flips.values())
        print(f"  M={M:<3}  delta={delta_test:.6f}  "
              f"flip@0.50={flips[0.50]:.6f}  flip@0.75={flips[0.75]:.6f}  "
              f"ECART={spread:.6f}")
