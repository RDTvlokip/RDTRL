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
                     s3_init=0.999999999666, r_tie=0.5, avec_mi_parcours=False,
                     r_autres_init=1e-6):
    """avec_mi_parcours=True renvoie AUSSI s3 a pas//2 -- necessaire au
    critere de TENDANCE (s3 decroit-il ?) plutot qu'un seuil absolu :
    pres de s3=0,5, la dynamique de ce jouet montre un plateau tres lent
    (verifie empiriquement : s3 encore a 0,500000 a 15000 pas, 0,499935 a
    40000, 0,498566 a 100000 pas, delta=0,05 -- bien au-dela du delta_c
    du jouet) -- un seuil absolu s3<0,5 a budget fixe n'est donc PAS fiable.

    r_autres_init=1e-6 (le defaut d'origine) rend les M categories de
    fond non-competitives des l'init (verifie le 17/09/2026 : elles
    bougent mais n'influencent jamais Z, 6 ordres de grandeur d'ecart) --
    passer une valeur plus grande (ex 0,01) pour un vrai test de
    competition, en gardant M*r_autres_init < 1."""
    poids3 = torch.tensor((1.0 - delta) / N, dtype=torch.float64)
    poids4 = torch.tensor((1.0 + delta) / N, dtype=torch.float64)
    p3, p4, q = construire_toy_m(M, delta, s3_init=s3_init, r_tie=r_tie, r_autres_init=r_autres_init)
    opt = torch.optim.Adam([p3, p4, q], lr=lr, eps=adam_eps)
    s3_mi = None
    for i in range(pas):
        j = objectif_toy_m(p3, p4, q, poids3, poids4)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if avec_mi_parcours and i == pas // 2:
            with torch.no_grad():
                s3_mi = torch.sigmoid(p3).item()
    with torch.no_grad():
        s3 = torch.sigmoid(p3).item()
        s4 = torch.sigmoid(p4).item()
        r_all = torch.softmax(q, dim=0)
        r3, r4 = r_all[0].item(), r_all[1].item()
        masse_autres = 1.0 - r3 - r4
        R4 = s4 * r4
    if avec_mi_parcours:
        return R4, s3, s4, r3, r4, masse_autres, s3_mi
    return R4, s3, s4, r3, r4, masse_autres


def bissecter_delta_c(M, lo=0.010, hi=0.020, tol=1e-4, pas=40000, lr=0.2, r_autres_init=1e-6):
    """lr=0.2 (au lieu de 0,05) -- teste empiriquement : a lr=0,05, la
    dynamique pres du point d'entropie (s3=0,5, l'attracteur "effondre"
    de CE jouet -- pas 1/27, contrairement au systeme complet, car il n'y
    a pas ici de 26 autres messages sur lesquels redistribuer la masse de
    l'emetteur) est si lente qu'un budget de 20000-150000 pas ne suffit
    pas a distinguer les deux issues (verifie : s3 encore a 0,500000-0,499935
    a 40000-100000 pas, delta=0,05, bien au-dela du delta_c du jouet). A
    lr=0,2, la separation devient nette en 40000 pas (verifie)."""
    def sature(delta):
        _, s3, *_ = entrainer_toy_m(M, delta, pas=pas, lr=lr, r_autres_init=r_autres_init)
        assert s3 > 0.9 or s3 < 0.6, \
            f"M={M} delta={delta}: s3={s3:.6f} ambigu (ni graduee ni effondree nettement)"
        return s3 < 0.6

    s_lo, s_hi = sature(lo), sature(hi)
    assert s_lo != s_hi, f"M={M}: lo/hi meme issue ({s_lo}/{s_hi}), elargir le bracket"
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if sature(mid) == s_hi:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def bissecter_flip_s3(M, delta, r_tie_init, lo=0.90, hi=0.999999, tol=1e-4, pas=40000, lr=0.2):
    def gradue(s3_init):
        """lr=0,2 + seuil a marge claire -- meme correction que
        bissecter_delta_c (voir sa docstring : lr=0,05 est trop lent
        pres de l'attracteur effondre de ce jouet, s3=0,5)."""
        _, s3f, *_ = entrainer_toy_m(M, delta, pas=pas, lr=lr, s3_init=s3_init, r_tie=r_tie_init)
        assert s3f > 0.9 or s3f < 0.6, \
            f"M={M} delta={delta} s3_init={s3_init}: s3f={s3f:.6f} ambigu"
        return s3f > 0.9

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
    Ms = (0, 8, 25)
    deltas_c = {}
    for M in Ms:
        dc = bissecter_delta_c(M, tol=3e-4, pas=40000, lr=0.2)
        deltas_c[M] = dc
        print(f"  M={M:<3}  delta_c={dc:.6f}   (jouet pur M=0: ~0,0187 ; systeme complet (M=25 vise): 0,013437)")

    print("\n=== 2) ecart des points de bascule entre R_init=0,50/0,75, a M croissant ===")
    print("    (delta fixe a 0,95*delta_c(M) pour rester comparable relativement au pli de CHAQUE M)")
    for M in Ms:
        dc = deltas_c[M]
        delta_test = 0.95 * dc
        flips = {}
        for r_tie in (0.50, 0.75):
            flip = bissecter_flip_s3(M, delta_test, r_tie, tol=3e-4, pas=40000, lr=0.2)
            flips[r_tie] = flip
        spread = max(flips.values()) - min(flips.values())
        print(f"  M={M:<3}  delta={delta_test:.6f}  "
              f"flip@0.50={flips[0.50]:.6f}  flip@0.75={flips[0.75]:.6f}  "
              f"ECART={spread:.6f}")
