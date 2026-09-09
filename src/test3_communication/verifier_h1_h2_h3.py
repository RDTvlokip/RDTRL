"""Trois hypotheses pour le creux non-monotone de 1-s[0,0] a pas=8000
(2.1,2.3,2.6,2.9,pic 3.3,2.9,2.5,2.2,2.0e-9), toutes testees plutot que
listees (consigne CLAUDE.md, 09/09/2026) :

H1 - artefact d'Adam (moment/variance adaptatifs pres d'un changement
     brutal de gradient) -> teste en rejouant la meme fenetre avec SGD
     pur (monter() utilise Adam partout, jamais de SGD reel jusqu'ici).
H2 - couplage par l'entropie (les 26 autres logits du referent 0
     bougent via la normalisation softmax) -> teste en regardant si la
     SOMME des logits ou la moyenne des logits non-gagnants bouge au
     meme moment.
H3 - rétroaction du recepteur (le creux coincide avec le moment ou
     r[0,0] bascule reellement) -> teste en imprimant r[0,0] a cote du
     logit du referent 0 aux memes checkpoints.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_idx5 import replay_idx5, BETA
from representable_atteignable_stable import objectif, activer, parametres

torch.set_printoptions(precision=15)

PAS_LISTE = (7000, 7500, 8000, 8500, 9000)


def monter_sgd(e, r, beta, pas, lr):
    activer(e, r)
    opt = torch.optim.SGD(parametres(e, r), lr=lr)
    for _ in range(pas):
        j, _ = objectif(e, r, beta)
        opt.zero_grad()
        (-j).backward()
        opt.step()


print("=== H1 : meme fenetre, SGD pur au lieu d'Adam ===")
for pas_suite in PAS_LISTE:
    e, r = replay_idx5(40000)
    with torch.no_grad():
        e.p[0][0, 0] += 24.0
    monter_sgd(e, r, BETA, pas_suite, lr=0.05)
    with torch.no_grad():
        row = e.p[0][0]
        s_ = torch.softmax(row, dim=0)
        top = row.topk(3)
    print(f"pas={pas_suite:6d}  top1={float(top.values[0]):.6f}  1-s[0,0]={1.0 - s_[0].item():.6e}")

print()
print("=== H2 : somme des logits du referent 0 (hors le gagnant), meme fenetre, Adam ===")
from representable_atteignable_stable import monter
for pas_suite in PAS_LISTE:
    e, r = replay_idx5(40000)
    with torch.no_grad():
        e.p[0][0, 0] += 24.0
    monter(e, r, BETA, pas_suite, lr=0.05)
    with torch.no_grad():
        row = e.p[0][0]
        somme_totale = row.sum().item()
        somme_hors_gagnant = (row.sum() - row[0]).item()
        s_ = torch.softmax(row, dim=0)
    print(f"pas={pas_suite:6d}  somme_totale={somme_totale:.6f}  somme_hors_gagnant={somme_hors_gagnant:.6f}  1-s[0,0]={1.0 - s_[0].item():.6e}")

print()
print("=== H3 : r[0,0] (recepteur) a cote du logit du referent 0, meme fenetre, Adam ===")
for pas_suite in PAS_LISTE:
    e, r = replay_idx5(40000)
    with torch.no_grad():
        e.p[0][0, 0] += 24.0
    monter(e, r, BETA, pas_suite, lr=0.05)
    with torch.no_grad():
        row = e.p[0][0]
        s_ = torch.softmax(row, dim=0)
        r_ = r.loi()
    print(f"pas={pas_suite:6d}  top1_logit={float(row.topk(1).values[0]):.6f}  1-s[0,0]={1.0 - s_[0].item():.6e}  r[0,0]={r_[0,0].item():.6f}")
