"""Suite immediate (agent-dipankar, 18/09/2026) du script de temps de
residence : mesure DIRECTEMENT le taux local lambda pres du
ralentissement (t=260-290, deja documente CARNET.md PERCEE) sur la
trajectoire SOUS a offset=1e-5 (la plus proche du seuil de bissection),
pour comparer a la pente du fit log (a=-194.652 pour bande=1e-3/pas=3000,
-204.902 pour bande=5e-4/pas=5000) SANS passer par le comptage en bande.

Si le rythme local mesure directement pres du ralentissement colle a
1/(|a|*ln10), la pente log s'explique bien par la geometrie DU COL
lui-meme (linearisation), pas par un artefact en amont (demarrage a
froid d'Adam, dynamique d'approche avant d'atteindre le voisinage).
"""

import sys
sys.path.insert(0, '.')
import numpy as np
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from verifier_sonde_bassin import fixer_s3, poids_delta

DELTA = 0.013
ADAM_EPS = 1e-10
LR = 0.05
SEUIL = 0.9726535
CIBLE_BANDE = 0.994300


def trace_verbose(cible_s3_init, pas_max):
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    fixer_s3(e, cible_s3_init)
    activer(e, r)
    poids = poids_delta(DELTA)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS)
    traj = []
    for t in range(pas_max):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            s3 = e.loi()[3, 10].item()
        traj.append(s3)
    return traj


if __name__ == "__main__":
    offset = 1e-5
    s3_init = SEUIL - offset
    print(f"=== Trajectoire verbose SOUS, offset={offset:.0e}, s3_init={s3_init:.7f} ===")
    traj = trace_verbose(s3_init, 350)

    print("\n  t     s3           |s3-0.9943|")
    for t in range(230, 320, 5):
        d = abs(traj[t] - CIBLE_BANDE)
        print(f"  {t:4d}  {traj[t]:.8f}  {d:.3e}")

    # vitesse |delta s3| par pas, cherche le minimum (quasi-immobile), comme
    # dans la section PERCEE du CARNET
    vitesses = [abs(traj[t+1] - traj[t]) for t in range(len(traj)-1)]
    t_min_vitesse = int(np.argmin(vitesses[200:320])) + 200
    print(f"\n  vitesse minimale (quasi-immobile) a t={t_min_vitesse}, "
          f"|Delta s3|={vitesses[t_min_vitesse]:.3e}, s3={traj[t_min_vitesse]:.8f}")

    # taux local : ajuste log|s3 - s3_lim| = -lambda*t + const sur une fenetre
    # APRES le point de vitesse minimale (phase d'evacuation qui suit)
    fen_debut = t_min_vitesse + 5
    fen_fin = min(fen_debut + 40, len(traj) - 1)
    ts = np.arange(fen_debut, fen_fin)
    ecarts = np.array([abs(traj[t] - CIBLE_BANDE) for t in ts])
    mask = ecarts > 1e-12
    ts_f, ecarts_f = ts[mask], ecarts[mask]
    if len(ts_f) >= 3:
        logy = np.log(ecarts_f)
        A = np.vstack([ts_f, np.ones_like(ts_f)]).T
        coef, *_ = np.linalg.lstsq(A, logy, rcond=None)
        lam_local = coef[0]
        print(f"\n  fenetre fit taux local : pas {fen_debut}-{fen_fin}")
        print(f"  lambda_local (croissance exponentielle de |s3-0.9943|) = {lam_local:.6f} /pas")
        print(f"  e-folding = {1/abs(lam_local):.2f} pas" if lam_local != 0 else "")
    else:
        print("\n  pas assez de points valides pour le fit local (ecarts trop petits / bruit float)")

    a_log_bande1 = -194.652
    a_log_bande2 = -204.902
    print(f"\n  comparaison : 1/(|a_log|*ln10) pour bande=1e-3/pas=3000 -> {1/(abs(a_log_bande1)*np.log(10)):.6f} /pas")
    print(f"  comparaison : 1/(|a_log|*ln10) pour bande=5e-4/pas=5000 -> {1/(abs(a_log_bande2)*np.log(10)):.6f} /pas")
