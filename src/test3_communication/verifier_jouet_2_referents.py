"""Ablation directe de H13 : si les 25 autres referents/messages sont la
DESTINATION de la masse qui s'echappe (l'effondrement atterrit sur 1/27,
pas 1/2), alors les retirer completement doit soit faire disparaitre
delta_c, soit le deplacer nettement, soit faire atterrir l'effondrement
sur 1/2 au lieu de 1/27.

Jouet autonome, sans les 27x27 matrices completes : deux emetteurs a 2
issues (message "10" vs "ailleurs", 1 seule categorie ailleurs au lieu
de 26), un recepteur a 2 issues (referent 3 vs referent 4, au lieu de
27 referents). Meme beta=0.02, meme normalisation /N=27 (on ne change
que le nombre de "passagers", pas l'echelle beta/N qui gouverne tout le
mecanisme).
"""

import torch
import math

BETA = 0.02
N = 27  # meme normalisation que le systeme complet -- seul le nombre
        # de categories "ailleurs" change, pas l'echelle beta/N


def construire_toy(delta, s3_init=0.999, s4_init=0.999, r_tie=0.5):
    p3 = torch.tensor([math.log(s3_init / (1 - s3_init))], dtype=torch.float64, requires_grad=True)
    p4 = torch.tensor([math.log(s4_init / (1 - s4_init))], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([math.log(r_tie / (1 - r_tie))], dtype=torch.float64, requires_grad=True)
    return p3, p4, q


def entropie_binaire(p):
    return -(p * torch.log(p.clamp_min(1e-300)) + (1 - p) * torch.log((1 - p).clamp_min(1e-300)))


def objectif_toy(p3, p4, q, poids3, poids4):
    s3 = torch.sigmoid(p3)
    s4 = torch.sigmoid(p4)
    r4 = torch.sigmoid(q)
    r3 = 1 - r4
    recompense = poids3 * s3 * r3 + poids4 * s4 * r4
    entropie = entropie_binaire(s3) + entropie_binaire(s4) + entropie_binaire(r4)
    return (recompense + (BETA / N) * entropie).sum()


def entrainer(delta, pas=40000, lr=0.05, adam_eps=1e-10):
    poids3 = torch.tensor((1.0 - delta) / N, dtype=torch.float64)
    poids4 = torch.tensor((1.0 + delta) / N, dtype=torch.float64)
    p3, p4, q = construire_toy(delta)
    opt = torch.optim.Adam([p3, p4, q], lr=lr, eps=adam_eps)
    for _ in range(pas):
        j = objectif_toy(p3, p4, q, poids3, poids4)
        opt.zero_grad()
        (-j).backward()
        opt.step()
    with torch.no_grad():
        s3 = torch.sigmoid(p3).item()
        s4 = torch.sigmoid(p4).item()
        r4 = torch.sigmoid(q).item()
        R4 = s4 * r4
    return R4, s3, s4, r4


if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    print("=== jouet a 2 referents (ablation des 25 autres lignes) ===")
    print("  effondrement predit si H13 juste : atterrit sur 1/2 = 0.5, pas 1/27")
    for delta in (0.0, 0.005, 0.010, 0.012, 0.013, 0.0134, 0.014, 0.02, 0.05, 0.1, 0.3, 1.0):
        R4, s3, s4, r4 = entrainer(delta)
        pred = 1.0 / (1.0 + math.exp(-2 * delta / BETA)) if delta > 0 else 0.5
        print(f"  delta={delta:<7}  R4={R4:.6f}  s3={s3:.6f}  s4={s4:.6f}  r4={r4:.6f}  "
              f"prediction_molle={pred:.6f}")
