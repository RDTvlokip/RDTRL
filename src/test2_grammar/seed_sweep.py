"""Balayage du coefficient d'entropie sur PLUSIEURS graines.

Motivation : le balayage a graine unique de rl_grammaire.py est contamine.
A beta = 0.08, la graine 0 reste sur une seule branche (24 modes, 95 % valide)
alors que les graines 1, 2 et 3 couvrent les deux branches (43-46 modes,
78-90 % valide). Autrement dit certaines graines dominent, sur LES DEUX axes,
des points que le tableau a graine unique presente comme meilleurs.

Une frontiere validite / diversite tracee sur une seule graine n'est donc pas
publiable. Ce script refait le balayage sur plusieurs graines et sauvegarde
moyenne et ecart-type par coefficient.
"""

import csv
import json
import os

import numpy as np
import torch

from grammar import Grammaire
from rl_grammar import (PolitiqueGRU, analyse_exacte, entrainer, fixer_graine,
                          DOSSIER_SORTIE)

COEFS = [0.0, 0.01, 0.02, 0.05, 0.08, 0.12, 0.2, 0.35]
GRAINES = [0, 1, 2]
EPISODES = 20000

if __name__ == "__main__":
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    grammaire = Grammaire(longue=False)
    lignes = []

    print(f"Balayage {len(COEFS)} coefficients x {len(GRAINES)} graines "
          f"x {EPISODES} episodes")
    print(f"{'coef':>6} {'graine':>7} {'valide%':>9} {'modes':>7} {'unifor%':>8} "
          f"{'sg%':>6} {'pl%':>6} {'P(nom|det)':>11} {'P(vb|nom)':>10}")

    for coef in COEFS:
        for graine in GRAINES:
            fixer_graine(graine)
            politique = PolitiqueGRU(grammaire.taille)
            entrainer(politique, grammaire, max_episodes=EPISODES,
                      type_recompense="graduee", coef_entropie=coef,
                      verbeux=False, etiquette=f"c{coef}g{graine}")
            ex = analyse_exacte(politique, grammaire)
            ligne = {
                "coef_entropie": coef,
                "graine": graine,
                "masse_valide_pct": ex["masse_valide_pct"],
                "modes_effectifs": ex["modes_effectifs"],
                "uniformite_pct": ex["uniformite_pct"],
                "sg_pct": ex["repartition_familles"]["sg"],
                "pl_pct": ex["repartition_familles"]["pl"],
                "p_nom_sachant_det": ex["moyenne_cond_det"],
                "p_verbe_sachant_nom": ex["moyenne_cond_nom"],
            }
            lignes.append(ligne)
            print(f"{coef:>6} {graine:>7} {ligne['masse_valide_pct']:>9.2f} "
                  f"{ligne['modes_effectifs']:>7.1f} {ligne['uniformite_pct']:>8.1f} "
                  f"{ligne['sg_pct']:>6.1f} {ligne['pl_pct']:>6.1f} "
                  f"{ligne['p_nom_sachant_det']:>11.3f} "
                  f"{ligne['p_verbe_sachant_nom']:>10.3f}")

    with open(os.path.join(DOSSIER_SORTIE, "balayage_graines.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)

    print()
    print("Synthese (moyenne +/- ecart-type sur les graines) :")
    print(f"{'coef':>6} {'valide%':>16} {'modes':>16} {'branches vues':>15}")
    synthese = []
    for coef in COEFS:
        groupe = [l for l in lignes if l["coef_entropie"] == coef]
        v = [l["masse_valide_pct"] for l in groupe]
        m = [l["modes_effectifs"] for l in groupe]
        deux_branches = sum(1 for l in groupe if min(l["sg_pct"], l["pl_pct"]) > 5)
        print(f"{coef:>6} {np.mean(v):>8.1f} +/-{np.std(v):>5.1f} "
              f"{np.mean(m):>8.1f} +/-{np.std(m):>5.1f} "
              f"{deux_branches:>10}/{len(groupe)}")
        synthese.append({"coef": coef, "valide_moy": round(float(np.mean(v)), 2),
                         "valide_ec": round(float(np.std(v)), 2),
                         "modes_moy": round(float(np.mean(m)), 2),
                         "modes_ec": round(float(np.std(m)), 2),
                         "graines_deux_branches": deux_branches, "n": len(groupe)})

    with open(os.path.join(DOSSIER_SORTIE, "balayage_graines.json"), "w",
              encoding="utf-8") as f:
        json.dump({"detail": lignes, "synthese": synthese}, f, indent=2, ensure_ascii=False)
    print(f"\nEcrit dans {DOSSIER_SORTIE}")
