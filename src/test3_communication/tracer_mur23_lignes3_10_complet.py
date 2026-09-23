"""Tour 58 (23/09/2026, VRAIE critique de dipankarsarkar) : enregistre, PAS
PAR PAS, l'etat complet des deux lignes en jeu dans le co-timing
s3 <-> kick recepteur, pour pouvoir faire la comparaison ALIGNEE SUR LES
EVENEMENTS qu'il demande (d3 - dbar a chacun des 16 evenements delta=0,
a cote des 15 evenements delta reel) sans relancer la simulation a
chaque nouvelle question.

Pourquoi un enregistrement complet plutot qu'un script d'analyse direct :
le tour 57 a montre qu'un pas apparie n'est pas un evenement apparie
(59989 tombait ENTRE deux kicks a delta=0). Toute analyse aligne sur les
evenements a besoin de la trace entiere, pas de points pre-selectionnes.

Enregistre sur [DEBUT, PAS_MAX) (apres construire_mur23, meme convention
de comptage que les scripts du tour 57) :
  - logits de l'emetteur, lignes 3 et 4 (27 messages chacune)
  - logits du recepteur, ligne 10 (27 referents)
  - gradient de -J sur ces trois lignes (ce qu'Adam recoit)
  - etat d'Adam (exp_avg, exp_avg_sq) sur ces trois lignes, APRES le pas

Usage : python tracer_mur23_lignes3_10_complet.py <delta>
Sortie : D:/tmp/rdtrl_tour58_trace_mur23_g77777_k3_delta<delta>_pas<DEBUT>-<PAS_MAX>.pt
(le nom porte graine, k, delta et fenetre -- cf. memoire
artefact-doit-porter-sa-provenance).
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

ADAM_EPS = 1e-10
LR = 0.05
DEBUT = 54000
PAS_MAX = 62000


def tracer(delta):
    torch.set_num_threads(1)
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + delta) / N
    poids[3] = (1.0 - delta) / N
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)
    p_e, p_r = e.p[0], r.p[0]

    n = PAS_MAX - DEBUT
    champs = ("e3", "e4", "r10", "g_e3", "g_e4", "g_r10",
              "m_e3", "m_r10", "v_e3", "v_e4", "v_r10")
    trace = {c: torch.empty((n, N), dtype=torch.float64) for c in champs}

    for pas in range(PAS_MAX):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        if pas >= DEBUT:
            i = pas - DEBUT
            trace["g_e3"][i] = p_e.grad[3].detach().clone()
            trace["g_e4"][i] = p_e.grad[4].detach().clone()
            trace["g_r10"][i] = p_r.grad[10].detach().clone()
        opt.step()
        if pas >= DEBUT:
            with torch.no_grad():
                trace["e3"][i] = p_e[3].clone()
                trace["e4"][i] = p_e[4].clone()
                trace["r10"][i] = p_r[10].clone()
                st_e, st_r = opt.state[p_e], opt.state[p_r]
                trace["m_e3"][i] = st_e["exp_avg"][3].clone()
                trace["m_r10"][i] = st_r["exp_avg"][10].clone()
                trace["v_e3"][i] = st_e["exp_avg_sq"][3].clone()
                trace["v_e4"][i] = st_e["exp_avg_sq"][4].clone()
                trace["v_r10"][i] = st_r["exp_avg_sq"][10].clone()
    trace["pas"] = torch.arange(DEBUT, PAS_MAX)
    trace["delta"] = delta
    return trace


if __name__ == "__main__":
    delta = float(sys.argv[1])
    trace = tracer(delta)
    chemin = (f"D:/tmp/rdtrl_tour58_trace_mur23_g77777_k3_delta{delta}"
              f"_pas{DEBUT}-{PAS_MAX}.pt")
    torch.save(trace, chemin)
    print(f"ecrit : {chemin}")
