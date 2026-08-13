"""
RDTRL — Test 3 : l'ecart max/appariee tombe-t-il avec beta, ou R cesse-t-il de le predire ?

Question de Dipankar Sarkar, 12/08/2026. Il fait remarquer que sa loi nulle par R
est plate — melangee sur p(R), la nulle appariee bouge de 0,00016 quand la mesure,
elle, bouge de 0,1355 — donc le probleme des trois bornes n'etait pas la reference
mais l'ensemble atteignable de la recherche. Le numerateur, pas le denominateur.

Sa reserve : son p(R) est le mien, donc le melange herite du calendrier de beta de
mon emetteur. R moyen monte de 24,05 a 25,25 quand beta va de 0,005 a 0,037, et le
haut de cette plage pousse p(R) vers la cellule qui rapporte le plus petit effet.

    « Does the effect size fall as you raise beta, or does R stop being the
      variable that predicts it ? »

Les deux sont testables, et les distinguer demande de conditionner. On mesure sur
des codes REELLEMENT EMERGENTS, pas sur un pire cas de recherche locale :

  - l'ecart observe max − appariee par beta ;
  - le meme ecart par R, toutes valeurs de beta confondues ;
  - et le croisement, qui est le seul a repondre : a R fixe, beta change-t-il
    encore quelque chose ?

Si l'ecart est une fonction de R seul, les colonnes du croisement sont plates et
tout l'effet de beta passe par le deplacement de p(R). Si beta agit au-dela, les
lignes bougent a R constant, et R n'est pas la bonne variable.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from grammaire3 import DOSSIER_SORTIE, N
from loi_nulle_longue import matrices_information_generale, statistiques
from representable_atteignable_stable import (EmetteurTabulaire, Recepteur,
                                              lire_code, monter)

torch.set_num_threads(int(os.environ.get("RDTRL_THREADS", "1")))


def une_serie(beta, graines, pas, generateur):
    runs = []
    for _ in range(graines):
        emetteur, recepteur = EmetteurTabulaire(generateur), Recepteur(generateur)
        recompense = monter(emetteur, recepteur, beta, pas)
        code = lire_code(emetteur)
        cm, ca, _ = statistiques(
            matrices_information_generale(np.asarray(code)[None, :]))
        runs.append({"beta": beta, "reward": recompense,
                     "R": int(len(np.unique(code))),
                     "max": float(cm[0]), "appariee": float(ca[0]),
                     "ecart": float(cm[0] - ca[0])})
    return runs


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="RDTRL — ecart par beta et par R")
    parseur.add_argument("--graines", type=int, default=30)
    parseur.add_argument("--pas", type=int, default=3000)
    parseur.add_argument("--betas", type=float, nargs="*",
                         default=[0.005, 0.010, 0.020, 0.030, 0.037])
    parseur.add_argument("--graine", type=int, default=0)
    args = parseur.parse_args()
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    generateur = np.random.default_rng(args.graine)

    print("=" * 78)
    print("L'ECART max - appariee : FONCTION DE BETA, OU DE R ?")
    print("=" * 78)
    print(f"\n  {args.graines} graines par beta, emetteur tabulaire, codes emergents.\n")

    tous = []
    print(f"  {'beta':>7}  {'E[R]':>7}  {'R moyen':>8}  {'max':>8}  {'appariee':>9}  "
          f"{'ecart':>16}")
    par_beta = []
    for beta in args.betas:
        runs = une_serie(beta, args.graines, args.pas, generateur)
        tous.extend(runs)
        e = np.array([r["ecart"] for r in runs])
        ligne = {"beta": beta, "n": len(runs),
                 "reward": float(np.mean([r["reward"] for r in runs])),
                 "R_moyen": float(np.mean([r["R"] for r in runs])),
                 "max": float(np.mean([r["max"] for r in runs])),
                 "appariee": float(np.mean([r["appariee"] for r in runs])),
                 "ecart": float(e.mean()),
                 "ecart_erreur": float(e.std(ddof=1) / np.sqrt(len(e)))}
        par_beta.append(ligne)
        print(f"  {beta:7.3f}  {ligne['reward']:7.4f}  {ligne['R_moyen']:8.2f}  "
              f"{ligne['max']:8.4f}  {ligne['appariee']:9.4f}  "
              f"{e.mean():8.4f} ± {ligne['ecart_erreur']:.4f}", flush=True)

    print("\n" + "-" * 78)
    print("LE MEME ECART, PAR R, TOUTES VALEURS DE BETA CONFONDUES")
    print("-" * 78)
    par_r = defaultdict(list)
    for r in tous:
        par_r[r["R"]].append(r["ecart"])
    print(f"  {'R':>3}  {'n':>4}  {'ecart moyen':>18}")
    lignes_r = []
    for R in sorted(par_r, reverse=True):
        e = np.array(par_r[R])
        err = float(e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 1 else float("nan")
        lignes_r.append({"R": R, "n": len(e), "ecart": float(e.mean()),
                         "erreur": err})
        print(f"  {R:>3}  {len(e):>4}  {e.mean():9.4f} ± {err:.4f}")

    print("\n" + "-" * 78)
    print("LE CROISEMENT — c'est lui qui repond")
    print("-" * 78)
    print("  A R fixe, beta change-t-il encore l'ecart ? Colonnes plates = non.\n")
    rs = sorted(par_r, reverse=True)
    print(f"  {'beta':>7}  " + "  ".join(f"{'R=' + str(R):>12}" for R in rs))
    croise = {}
    for beta in args.betas:
        cellules = []
        for R in rs:
            e = [x["ecart"] for x in tous if x["beta"] == beta and x["R"] == R]
            croise[f"{beta}_{R}"] = {"n": len(e),
                                     "ecart": float(np.mean(e)) if e else None}
            cellules.append(f"{np.mean(e):.4f} ({len(e)})" if e else "     —      ")
        print(f"  {beta:7.3f}  " + "  ".join(f"{c:>12}" for c in cellules))

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    beta_v = np.array([r["beta"] for r in tous])
    r_v = np.array([r["R"] for r in tous], dtype=float)
    e_v = np.array([r["ecart"] for r in tous])
    print(f"  correlation ecart / R    : {np.corrcoef(r_v, e_v)[0, 1]:+.3f}")
    print(f"  correlation ecart / beta : {np.corrcoef(beta_v, e_v)[0, 1]:+.3f}")
    # residus apres retrait de la moyenne de sa cellule R : beta explique-t-il ce
    # qui reste ? Si oui, R n'est pas la variable suffisante.
    moyennes = {R: np.mean(par_r[R]) for R in par_r}
    residus = np.array([r["ecart"] - moyennes[r["R"]] for r in tous])
    print(f"  correlation residu / beta apres conditionnement sur R : "
          f"{np.corrcoef(beta_v, residus)[0, 1]:+.3f}")
    print(f"  ecart-type de l'ecart : total {e_v.std():.4f}, "
          f"residuel apres R {residus.std():.4f} "
          f"({100 * (1 - residus.var() / e_v.var()):.0f} % de variance expliquee par R)")

    rapport = {"graines": args.graines, "pas": args.pas, "graine": args.graine,
               "par_beta": par_beta, "par_R": lignes_r, "croisement": croise,
               "runs": tous}
    nom = f"effet_par_beta_{args.graines}graines_g{args.graine}.json"
    with open(os.path.join(DOSSIER_SORTIE, nom), "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)
    print(f"\nEcrit dans {DOSSIER_SORTIE} sous {nom}")
