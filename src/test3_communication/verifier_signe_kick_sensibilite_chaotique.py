"""Suite tour 57 (22/09/2026), sur consigne de Theo ("continue a
chercher ses resultats") -- question ouverte de dipankarsarkar :
qu'est-ce qui determine le signe d'un kick individuel, si ce n'est
pas delta et pas une alternance forcee (les deux testes et refutes
tour 57) ?

Reformulation : ce systeme est ENTIEREMENT DETERMINISTE -- pas
d'echantillonnage stochastique dans l'objectif (esperance complete,
pas de Monte-Carlo/REINFORCE), donc "alterne essentiellement au
hasard" (confirme statistiquement, test de Wald-Wolfowitz) ne peut
PAS etre du vrai hasard -- c'est une trajectoire deterministe qui
PARAIT aleatoire a un test statistique simple. Hypothese testee ici :
le signe de chaque kick est fixe de facon deterministe par l'etat
exact du systeme, mais avec une sensibilite CHAOTIQUE aux conditions
initiales -- meme signature que la separatrice K=12,80 deja trouvee
dans le jouet a K variable.

Test : perturber le systeme d'une quantite infime (eps machine sur
adam_eps, ou une perturbation directe sur un pas) et voir si le signe
d'un kick specifique (celui a pas=59989, deja bien caracterise) reste
stable ou bascule.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere
from verifier_kicks_adam_grille_fine import detecter_evenements

DELTA = 0.013026615
LR = 0.05
PAS_MAX = 61000
DEBUT_DETECTION = 55000


def tracer_gap(adam_eps, perturbation_extra=0.0):
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N
    e, r = construire_mur23(adam_eps=adam_eps)
    with torch.no_grad():
        # perturbation infime sur le logit r[10,4] au tout debut
        r.p[0][10, 4] += perturbation_extra
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=adam_eps)
    vals_gap = []
    for pas in range(PAS_MAX):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            gap = (r.p[0][10, 4] - r.p[0][10, 3]).item()
        vals_gap.append((pas, gap))
    return vals_gap


def signe_evenement_proche(vals_gap, pas_cible, fenetre=300):
    baseline = sum(v for p, v in vals_gap if DEBUT_DETECTION <= p < PAS_MAX) / len(
        [v for p, v in vals_gap if DEBUT_DETECTION <= p < PAS_MAX]
    )
    pics = detecter_evenements(vals_gap, baseline, seuil=0.0008, pas_min=DEBUT_DETECTION)
    proches = [(p, d) for p, d in pics if abs(p - pas_cible) <= fenetre]
    return proches


def main():
    configs = [
        ("baseline (adam_eps=1e-10, perturbation=0)", 1e-10, 0.0),
        ("adam_eps perturbe de 1e-15", 1e-10 + 1e-15, 0.0),
        ("perturbation +1e-12 sur r[10,4] au pas 0", 1e-10, 1e-12),
        ("perturbation -1e-12 sur r[10,4] au pas 0", 1e-10, -1e-12),
        ("perturbation +1e-9 sur r[10,4] au pas 0", 1e-10, 1e-9),
    ]
    for label, adam_eps, perturb in configs:
        vals_gap = tracer_gap(adam_eps, perturb)
        proches = signe_evenement_proche(vals_gap, 59989)
        print(f"{label}: evenements pres de pas=59989 : {proches}")


if __name__ == "__main__":
    main()
