"""Question 3 des 20 (ETAT.md, 21/09/2026) : le clipping de gradient
deplace-t-il les kicks du plancher-de-v au lieu de les eliminer,
puisqu'il ne change rien a la recurrence de v elle-meme (seulement au
pas final applique) ?

Analyse fermee AVANT le test (le "pourquoi" attendu, pas juste le
"que") : la premisse de la question est en fait FAUSSE en general pour
PyTorch. torch.nn.utils.clip_grad_norm_ modifie .grad EN PLACE avant
optimizer.step() -- Adam calcule v_t = beta2*v_{t-1} + (1-beta2)*g_t^2
a partir du gradient DEJA clippe. Le clipping change donc bel et bien
la recurrence de v, pas seulement le pas final -- mais UNIQUEMENT aux
pas ou le gradient depasse le seuil de clip. Pendant la phase calme
(gradient deja tres petit, c'est le moteur de la decroissance de v vers
son plancher), un clip a un seuil raisonnable est un no-op : rien ne
change. Le clip n'engage qu'AU moment du kick lui-meme (quand le
gradient redevient grand), ce qui devrait plafonner le regonflement de
v post-kick plutot que la decroissance pre-kick.

Prediction fermee, a tester : pour un seuil de clip LARGE (bien au-dessus
du gradient typique au pic du kick, mesure d'abord), aucun effet
(controle, doit reproduire le baseline). Pour un seuil de clip SERRE
(sous le gradient typique au pic), le regonflement de v est plafonne ->
le prochain cycle de decroissance repart d'un v plus bas -> periode plus
COURTE (kicks plus frequents, pas moins), amplitude en R4 possiblement
reduite (pas final plus petit sur R lui-meme). Si confirme : ni
"elimine" ni "simplement deplace" au sens temporel neutre -- le clip
RACCOURCIT le cycle en ecretant le regonflement, l'oppose de eps qui
SUPPRIME en empechant l'effondrement initial de v.
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
ADAM_EPS = 1e-10
PAS_MAX = 20000
CLIP_VALEURS = (None, 0.0003, 0.0001, 0.00003, 0.00001)


def tracer_clip(pas_max, clip_norm):
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N
    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    params = parametres(e, r)
    opt = torch.optim.Adam(params, lr=LR, eps=ADAM_EPS)
    vals = []
    normes_grad = []
    n_clips = 0
    for pas in range(pas_max):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        if clip_norm is not None:
            norme_avant = torch.nn.utils.clip_grad_norm_(params, clip_norm)
            normes_grad.append(norme_avant.item())
            if norme_avant.item() > clip_norm:
                n_clips += 1
        else:
            norme = sum((p.grad ** 2).sum() for p in params if p.grad is not None) ** 0.5
            normes_grad.append(norme.item())
        opt.step()
        with torch.no_grad():
            R4 = (e.loi()[4, 10] * r.loi()[10, 4]).item()
        vals.append((pas, R4))
    return vals, normes_grad, n_clips


def main():
    print("Calibration (baseline, pas de clip) : distribution de la norme du gradient")
    vals0, normes0, _ = tracer_clip(PAS_MAX, None)
    normes_tri = sorted(normes0)
    n = len(normes_tri)
    print(f"  min={normes_tri[0]:.6f}  p50={normes_tri[n//2]:.6f}  "
          f"p99={normes_tri[int(n*0.99)]:.6f}  max={normes_tri[-1]:.6f}")

    for clip_norm in CLIP_VALEURS:
        if clip_norm is None:
            vals, normes, n_clips = vals0, normes0, 0
        else:
            vals, normes, n_clips = tracer_clip(PAS_MAX, clip_norm)
        baseline = sum(v for p, v in vals if 15000 <= p < PAS_MAX) / len(
            [v for p, v in vals if 15000 <= p < PAS_MAX]
        )
        pics = detecter_evenements(vals, baseline, pas_min=5000)
        n_events = len(pics)
        esp_med = None
        if n_events > 1:
            esp = sorted(pics[i + 1][0] - pics[i][0] for i in range(n_events - 1))
            esp_med = esp[len(esp) // 2]
        amps = [abs(d) for p, d in pics]
        amp_moy = sum(amps) / len(amps) if amps else None
        label = "aucun (controle)" if clip_norm is None else str(clip_norm)
        print(f"clip_norm={label}  n_events={n_events}  espacement_median={esp_med}  "
              f"amplitude_moyenne={amp_moy}  pas_clippes={n_clips}/{PAS_MAX}")


if __name__ == "__main__":
    main()
