"""Test precommis propose par un agent-dipankar (18/09/2026) pour
distinguer deux explications du rebond non-monotone trouve dans
verifier_point_fixe_jouet_m.py (M=25, l'ecart au point fixe algebrique
rebondit de ~2 ordres de grandeur a pas=15000 apres etre descendu) :

  (a) H15 -- artefact du second moment d'Adam (beta2=0,999), deja
      confirme sur le systeme complet a 27 referents (optimiseur
      hybride, tour 52).
  (b) un "fantome"/bottleneck deterministe du pli lui-meme (forme
      normale x(t)=x0/(1+k*x0*t), decroissance en 1/t, pas
      exponentielle, sans aucun rapport avec Adam) -- existe aussi
      (confirme theoriquement par l'agent), possible cause alternative
      ou concurrente.

Test : faire varier beta2 (la memoire du second moment d'Adam) et
regarder si la recurrence/frequence des rebonds en depend. Un artefact
purement deterministe (b) ne devrait PAS changer avec beta2 ; un
artefact Adam (a) devrait.
"""

import sys
sys.path.insert(0, '.')
import torch

from verifier_jouet_n_variable import construire_toy_m, objectif_toy_m, N
from verifier_point_fixe_jouet_m import point_fixe

M = 25
R_AUTRES_INIT = 0.01
DELTA = 0.95 * 0.018516


def relax(beta2, pas=20000, lr=0.2):
    poids3 = torch.tensor((1.0 - DELTA) / N, dtype=torch.float64)
    poids4 = torch.tensor((1.0 + DELTA) / N, dtype=torch.float64)
    p3, p4, q = construire_toy_m(M, DELTA, r_autres_init=R_AUTRES_INIT)
    opt = torch.optim.Adam([p3, p4, q], lr=lr, betas=(0.9, beta2), eps=1e-10)
    _, _, r3f, _, _ = point_fixe(M, DELTA)
    ecarts = []
    for i in range(pas):
        j = objectif_toy_m(p3, p4, q, poids3, poids4)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            r3 = torch.softmax(q, dim=0)[0].item()
        ecarts.append(abs(r3 - r3f))
    return ecarts, r3f


if __name__ == "__main__":
    for beta2 in (0.999, 0.9):
        ec, r3f = relax(beta2)
        print(f"=== beta2={beta2}  memoire=1/(1-beta2)={1/(1-beta2):.0f} pas  (r3*={r3f:.6f}) ===")
        for t in range(0, len(ec), 1000):
            print(f"  pas={t:6d}  ecart={ec[t]:.4e}")

    print("\nConclusion attendue si H15 : beta2=0,9 (memoire courte) montre des")
    print("oscillations PLUS frequentes que beta2=0,999, pas de longues periodes")
    print("stables -- confirme le 18/09/2026 (voir CARNET.md fin de §7.65).")
