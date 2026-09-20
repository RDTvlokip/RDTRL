"""Tour 53 (20/09/2026) : DEUXIEME correction d'un biais d'echantillonnage
sur les excursions de R sous Adam complet, trouvee par un agent-dipankar
puis verifiee independamment -- le meme bug que la premiere retractation
("cycle limite periodique" -> "bruit frequent toutes les 10000-30000
pas") existait encore, UN NIVEAU PLUS BAS, dans la grille utilisee pour
cette premiere correction elle-meme (500-1000 pas, encore trop grossiere).

A grille 1 (aucun sous-echantillonnage), R4 ne fait PAS une excursion
de magnitude variable toutes les 10000-30000 pas : il fait un "kick" de
magnitude QUASI CONSTANTE (~0.0037, CV~5.6% rapporte par l'agent,
confirme independamment sur un segment plus court dans ce script)
toutes les ~460-500 pas. Le "bruit distribue en continu, borne, sans
queue lourde" mesure a grille 500-1000 etait un artefact d'aliasing
(un pas d'echantillonnage comparable a la periode du kick lui-meme).

Detection d'evenement : seuil sur |R4-baseline|, fusion des pas
consecutifs qui depassent le seuil (a l'interieur d'un meme "sonnement"
/ ringing) separes de moins de GAP_FUSION pas, pic pris comme
l'extremum a l'interieur de chaque evenement fusionne.

Mecanisme partiel propose par l'agent (PAS encore verifie ici) :
oscillateur de relaxation par plancher numerique de exp_avg_sq (v) --
v decroit vers un plancher pendant la phase calme, une petite
perturbation de gradient contre un historique de variance minuscule
produit un pas normalise demesure (lr*m/sqrt(v) explose), le gradient
du kick regonfle v, R sonne en retour vers la baseline. Voir
CARNET.md fin de 7.65/8ter pour le detail complet et le statut de
verification de chaque affirmation.
"""

import sys
sys.path.insert(0, '.')
import torch

from replay_mur23_referent3 import construire_mur23, BETA
from representable_atteignable_stable import N, activer, parametres
from verifier_prior_asymetrique import objectif_pondere

DELTA = 0.013026615
LR = 0.05
ADAM_EPS = 1e-10
SEUIL = 0.0015
GAP_FUSION = 50


def tracer(pas_max, betas=(0.9, 0.999)):
    poids = torch.full((N,), 1.0 / N, dtype=torch.float64)
    poids[4] = (1.0 + DELTA) / N
    poids[3] = (1.0 - DELTA) / N

    e, r = construire_mur23(adam_eps=ADAM_EPS)
    activer(e, r)
    opt = torch.optim.Adam(parametres(e, r), lr=LR, eps=ADAM_EPS, betas=betas)

    vals = []
    for pas in range(pas_max):
        j, _ = objectif_pondere(e, r, BETA, poids)
        opt.zero_grad()
        (-j).backward()
        opt.step()
        with torch.no_grad():
            R4 = (e.loi()[4, 10] * r.loi()[10, 4]).item()
        vals.append((pas, R4))
    return vals


def detecter_evenements(vals, baseline, seuil=SEUIL, gap_fusion=GAP_FUSION, pas_min=5500):
    crossings = [(p, v - baseline) for p, v in vals if p >= pas_min and abs(v - baseline) > seuil]
    if not crossings:
        return []
    evenements = [[crossings[0]]]
    for p, d in crossings[1:]:
        if p - evenements[-1][-1][0] <= gap_fusion:
            evenements[-1].append((p, d))
        else:
            evenements.append([(p, d)])
    pics = []
    for ev in evenements:
        pic = max(ev, key=lambda t: abs(t[1]))
        pics.append(pic)
    return pics


def main():
    torch.set_printoptions(precision=6)
    PAS_MAX = 60000
    vals = tracer(PAS_MAX)
    baseline = sum(v for p, v in vals if 20000 <= p < 40000) / len([v for p, v in vals if 20000 <= p < 40000])
    print(f"baseline = {baseline:.10f}")

    pics = detecter_evenements(vals, baseline)
    print(f"n evenements (grille 1, [5500,{PAS_MAX})) = {len(pics)}")
    if len(pics) > 1:
        espacements = [pics[i + 1][0] - pics[i][0] for i in range(len(pics) - 1)]
        espacements.sort()
        mediane = espacements[len(espacements) // 2]
        print(f"espacement median entre evenements = {mediane}")
    amplitudes = [abs(d) for p, d in pics]
    if amplitudes:
        moyenne = sum(amplitudes) / len(amplitudes)
        ecart_type = (sum((a - moyenne) ** 2 for a in amplitudes) / len(amplitudes)) ** 0.5
        print(f"amplitude moyenne = {moyenne:.6f}  ecart-type = {ecart_type:.6f}  CV = {100*ecart_type/moyenne:.2f}%")
        print(f"amplitude min/max = {min(amplitudes):.6f} / {max(amplitudes):.6f}")


if __name__ == "__main__":
    main()
