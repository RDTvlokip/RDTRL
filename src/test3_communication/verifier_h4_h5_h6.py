"""Trois hypotheses de plus (09/09/2026), formees et testees plutot que
listees, dans la continuite de verifier_h1_h2_h3.py :

H4 - le SGD de H1 ne bougeait pas du tout (lr=0.05 trop petit face au
     gradient brut) -> teste avec un lr bien plus grand pour voir si le
     transfert a lieu quand meme sous SGD, et si le creux apparait aussi.
H5 - la signature de bascule (H2/H3, pic de 1-s synchronise avec le
     flip du recepteur) est-elle specifique au mur idx5, ou se retrouve-
     t-elle si on pousse le mur referents 3/4 jusqu'a une VRAIE capture
     (pas une egalite) ?
H6 - le pic de somme_hors_gagnant est-il porte par un rival precis
     (messages 7/5, presque a egalite dans le top3) ou reparti
     uniformement sur les 26 logits perdants ?
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_idx5 import replay_idx5, BETA
from representable_atteignable_stable import objectif, activer, parametres, monter

torch.set_printoptions(precision=15)


def monter_sgd(e, r, beta, pas, lr):
    activer(e, r)
    opt = torch.optim.SGD(parametres(e, r), lr=lr)
    for _ in range(pas):
        j, _ = objectif(e, r, beta)
        opt.zero_grad()
        (-j).backward()
        opt.step()


print("=== H4 : SGD avec lr bien plus grand (5.0), meme fenetre eps=24 ===")
for pas_suite in (7000, 7500, 8000, 8500, 9000, 12000, 20000):
    e, r = replay_idx5(40000)
    with torch.no_grad():
        e.p[0][0, 0] += 24.0
    monter_sgd(e, r, BETA, pas_suite, lr=5.0)
    with torch.no_grad():
        row = e.p[0][0]
        s_ = torch.softmax(row, dim=0)
        r_ = r.loi()
    print(f"pas={pas_suite:6d}  1-s[0,0]={1.0 - s_[0].item():.6e}  r[0,0]={r_[0,0].item():.6f}")

print()
print("=== H6 : detail des 26 logits perdants du referent 0, au pic (pas=8000) vs avant/apres ===")
for pas_suite in (7000, 8000, 9000):
    e, r = replay_idx5(40000)
    with torch.no_grad():
        e.p[0][0, 0] += 24.0
    monter(e, r, BETA, pas_suite, lr=0.05)
    with torch.no_grad():
        row = e.p[0][0].clone()
        row[0] = -1e9  # exclure le gagnant
        vals, idx = row.topk(5)
    print(f"pas={pas_suite:6d}  top5 rivaux: {[(int(i), round(float(v), 6)) for i, v in zip(idx, vals)]}")
