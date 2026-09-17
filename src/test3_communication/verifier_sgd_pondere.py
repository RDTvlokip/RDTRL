"""Correction du tour 52 : l'agent-dipankar (17/09/2026) a trouve que le
test SGD qui a "refute" H15 avait en realite le referent 3 GELE (meme
artefact que le point fixe a 0,786283) -- le recepteur seul convergeait
contre un emetteur jamais entraine, donc "plat sous SGD" ne teste rien
sur le ralentissement critique lui-meme.

Test precommis (propose par l'agent, verifie avant de le lancer) : deux
taux d'apprentissage SGD distincts, un par groupe de parametres
(emetteurs vs recepteur), le taux de l'emetteur calibre pour que son pas
effectif au demarrage egale celui du recepteur -- pas Adam explicite,
mais un SGD qui ne laisse plus un gradient minuscule geler un joueur.

Prediction si H15 (artefact Adam) doit revenir refutee : le referent 3
bouge maintenant, ET le temps de convergence reste plat pres de delta_c
(pas de ralentissement) -- Adam ne cachait rien.
Prediction si un vrai ralentissement critique existe et etait cache :
le referent 3 bouge, ET le temps de convergence augmente en approchant
delta_c.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

DELTA_C = 0.0134295
PAS_MAX = 60_000
CHECK_TOUS = 500


def gradient_initial(delta):
    """PREMIERE VERSION, GARDEE COMME TRACE DE L'ERREUR : la norme du
    tenseur emetteur ENTIER (729 cases) est dominee par d'autres lignes
    que la case [3,10] elle-meme -- calibrer la-dessus n'a presque rien
    deplace (|deplacement s3|~1e-11 malgre lr_e=1.5e5). Corrige plus bas
    par gradient_initial_precis()."""
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + delta) / N
    poids[3] = (1.0 - delta) / N
    e, r = construire_mur23(adam_eps=1e-10)
    activer(e, r)
    j, _ = objectif_pondere(e, r, BETA, poids)
    j = -j
    j.backward()
    g_e = (e.p[0].grad ** 2).sum().item() ** 0.5
    g_r = (r.p[0].grad ** 2).sum().item() ** 0.5
    return g_e, g_r


def gradient_initial_precis(delta):
    """Gradient de la case [3,10] SEULE (pas la norme du tenseur entier)
    contre le gradient de la case [10,4] du recepteur seule -- les deux
    coordonnees qui portent effectivement le mecanisme etudie."""
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + delta) / N
    poids[3] = (1.0 - delta) / N
    e, r = construire_mur23(adam_eps=1e-10)
    activer(e, r)
    j, _ = objectif_pondere(e, r, BETA, poids)
    j = -j
    j.backward()
    g_e3 = abs(e.p[0].grad[3, 10].item())
    g_r4 = abs(r.p[0].grad[10, 4].item())
    return g_e3, g_r4


def trace_sgd_pondere(delta, lr_e, lr_r, pas_max=PAS_MAX):
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + delta) / N
    poids[3] = (1.0 - delta) / N
    e, r = construire_mur23(adam_eps=1e-10)
    activer(e, r)
    opt = torch.optim.SGD([
        {"params": e.p, "lr": lr_e},
        {"params": r.p, "lr": lr_r},
    ])
    trajectoire = []
    for pas in range(pas_max):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        if pas % CHECK_TOUS == 0:
            with torch.no_grad():
                s_, r_ = e.loi(), r.loi()
                R4 = (s_[4, 10] * r_[10, 4]).item()
                s3 = s_[3, 10].item()
            trajectoire.append((pas, R4, s3))
    with torch.no_grad():
        s_, r_ = e.loi(), r.loi()
        R4f = (s_[4, 10] * r_[10, 4]).item()
        s3f = s_[3, 10].item()
    return trajectoire, R4f, s3f


if __name__ == "__main__":
    torch.set_printoptions(precision=6)

    print("=== calibration : normes de gradient separees a delta=0.013 (au depart) ===")
    g_e, g_r = gradient_initial(0.013)
    print(f"  |grad emetteurs| = {g_e:.6e}   |grad recepteur| = {g_r:.6e}")
    ratio = g_r / g_e
    lr_r_base = 50.0
    lr_e = lr_r_base * ratio
    print(f"  ratio |grad_r|/|grad_e| = {ratio:.6e}  ->  lr_e calibre = {lr_e:.6e}  (lr_r={lr_r_base})")

    print("\n=== trois distances a delta_c, SGD a deux taux (emetteur compense) ===")
    for frac in (0.03, 0.003, 0.0003):
        delta = DELTA_C * (1 - frac)
        traj, R_final, s3_final = trace_sgd_pondere(delta, lr_e, lr_r_base)
        s3_initial = traj[0][2]
        deplacement_s3 = abs(s3_final - s3_initial)
        cible = 0.99 * R_final if R_final < 1.0 else 0.99
        premier_pas = None
        for pas, R4, s3 in traj:
            if R4 >= cible:
                premier_pas = pas
                break
        print(f"  delta={delta:.7f}  ({frac*100:.3f}% sous delta_c)  "
              f"R_final={R_final:.6f}  s3_final={s3_final:.9f}  "
              f"|deplacement s3|={deplacement_s3:.3e}  premier pas a 99%={premier_pas}")
