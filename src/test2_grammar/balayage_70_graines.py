"""Le choix de branche est-il equilibre ? 70 graines a beta fixe.

dipankarsarkar a raison sur deux points et je les corrige ici.

1. Mes "15 singulier / 9 pluriel sur 24 runs" ne sont pas 24 tirages. Ce sont
   3 graines x 8 valeurs de beta, et dans le regime d'effondrement la branche est
   decidee par la GRAINE, pas par beta : graine 0 pluriel, graines 1 et 2
   singulier, a tous les beta <= 0.05. J'avais donc 3 tirages, pas 24, et le
   docstring de balayage_graines.py fait deja cet argument un niveau plus bas.

2. Separer 2/3 de 1/2 demande environ 70 runs (puissance 80 %, alpha 5 %).

Ce script fait les 70 tirages a beta constant, une seule condition.

Prediction a departager, enregistree avant de lancer :
  - la mienne : le signal d'ordre 1 au nom favorise le singulier (+0.0167), donc
    le singulier sort plus souvent que 1/2 ;
  - la sienne : le coin pluriel a un plafond sans couplage de 24 contre 12, donc
    il vaut beta*ln2 de plus a recompense egale, donc le pluriel sort plus
    souvent que 1/2.
Les deux effets existent et sont de signes opposes. C'est le comptage qui dit
lequel domine, et a beta = 0 son terme disparait alors que le mien reste.
"""

import argparse
import json
import os
from collections import Counter

# Un GRU minuscule en lot de 1 ne tire aucun profit du multithread : les threads
# se marchent dessus. Un thread par processus, et on parallelise par tranche de
# graines a la place.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

from grammaire import Grammaire
from rl_grammaire import (PolitiqueGRU, analyse_exacte, entrainer, fixer_graine,
                          DOSSIER_SORTIE)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    c = 1 / (1 + z * z / n)
    centre = c * (ph + z * z / (2 * n))
    demi = c * z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return (round(float(centre - demi), 4), round(float(centre + demi), 4))


def binomial_bilateral(k, n, p=0.5):
    from math import comb
    pk = comb(n, k) * p ** k * (1 - p) ** (n - k)
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i)
                     for i in range(n + 1)
                     if comb(n, i) * p ** i * (1 - p) ** (n - i) <= pk + 1e-15))


def synthese(lignes, beta):
    """Statistique de branche sur l'ensemble des graines, tranches fusionnees."""
    compte = Counter(l["branche"] for l in lignes)
    effondres = [l for l in lignes if l["branche"] in ("sg", "pl")]
    n = len(effondres)
    k = sum(1 for l in effondres if l["branche"] == "sg")
    print(f"\n  beta = {beta}, {len(lignes)} graines")
    print(f"  branches : {dict(compte)}")
    if not n:
        return {}
    print(f"  effondrees sur une seule famille : {n} | {k} singulier, {n - k} pluriel")
    print(f"  proportion singulier : {k/n:.4f}")
    print(f"  Wilson 95 %          : {wilson(k, n)}")
    print(f"  binomial vs 1/2      : p = {binomial_bilateral(k, n):.4f}")
    print(f"  binomial vs 2/3      : p = {binomial_bilateral(k, n, 2/3):.4f}")
    for br in ("sg", "pl"):
        m = [l["modes_effectifs"] for l in effondres if l["branche"] == br]
        if m:
            print(f"  modes, branche {br} : {np.mean(m):6.2f} +/- {np.std(m):5.2f} "
                  f"| min {min(m):5.1f} max {max(m):5.1f} | n = {len(m)}")
    return {"n_total": len(lignes), "n_effondres": n, "k_singulier": k,
            "wilson95": wilson(k, n),
            "p_contre_moitie": binomial_bilateral(k, n),
            "p_contre_deux_tiers": binomial_bilateral(k, n, 2/3),
            "branches": dict(compte)}


def fusionner(beta):
    """Relit toutes les tranches ecrites par les processus paralleles."""
    import glob
    motif = os.path.join(DOSSIER_SORTIE, f"balayage_70_graines_b{beta}_*.json")
    lignes = []
    for chemin in sorted(glob.glob(motif)):
        with open(chemin, encoding="utf-8") as f:
            lignes.extend(json.load(f)["detail"])
    lignes.sort(key=lambda l: l["graine"])
    vues = {l["graine"] for l in lignes}
    manquantes = set(range(max(vues) + 1)) - vues if vues else set()
    if manquantes:
        print(f"  ATTENTION : graines manquantes {sorted(manquantes)}")
    res = {"detail": lignes, "synthese": synthese(lignes, beta)}
    chemin = os.path.join(DOSSIER_SORTIE, f"balayage_70_graines_b{beta}_fusion.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n  Fusion ecrite dans {chemin}")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fusion", action="store_true",
                   help="relit les tranches et calcule la statistique globale")
    p.add_argument("--beta", type=float, default=0.02)
    p.add_argument("--graines", type=int, default=70)
    p.add_argument("--debut", type=int, default=0,
                   help="premiere graine de la tranche (parallelisation)")
    p.add_argument("--fin", type=int, default=None,
                   help="derniere graine exclue ; par defaut --graines")
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--chemin", choices=["float32", "float64"], default="float32",
                   help="ou se fait la soustraction recompense - baseline")
    args = p.parse_args()
    fin = args.graines if args.fin is None else args.fin

    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    if args.fusion:
        fusionner(args.beta)
        return
    dossier_poids = os.path.join(DOSSIER_SORTIE, "politiques_70")
    os.makedirs(dossier_poids, exist_ok=True)
    g = Grammaire(longue=False)
    lignes = []

    print(f"graines {args.debut} a {fin - 1} a beta = {args.beta}, "
          f"{args.episodes} episodes")
    print(f"{'graine':>7} {'valide%':>9} {'modes':>7} {'sg%':>7} {'pl%':>7} {'branche':>8}")
    for graine in range(args.debut, fin):
        fixer_graine(graine)
        politique = PolitiqueGRU(g.taille)
        entrainer(politique, g, max_episodes=args.episodes, type_recompense="graduee",
                  coef_entropie=args.beta, verbeux=False, etiquette=f"g{graine}",
                  chemin_avantage=args.chemin)
        ex = analyse_exacte(politique, g)
        sg, pl = ex["repartition_familles"]["sg"], ex["repartition_familles"]["pl"]
        branche = "sg" if sg > pl else ("pl" if pl > sg else "ex aequo")
        if min(sg, pl) > 5:
            branche = "les deux"
        # On garde TOUT ce qui porte de la masse. La version precedente ne
        # sauvait que moyenne_cond_det, qui est une moyenne non ponderee sur les
        # six determinants et vaut donc (determinants emis)/6, pas un accord.
        # masse_par_determinant existait un cran au-dessus et n'arrivait pas
        # jusqu'ici : meme faute que saturation_pct, deux fonctions plus haut.
        lignes.append({"graine": graine, "beta": args.beta,
                       "masse_valide_pct": ex["masse_valide_pct"],
                       "modes_effectifs": ex["modes_effectifs"],
                       "sg_pct": sg, "pl_pct": pl, "branche": branche,
                       "p_nom_sachant_det": ex["moyenne_cond_det"],
                       "cond_det_pondere": ex["cond_det_pondere"],
                       "information_mutuelle_det_nom_bits":
                           ex["information_mutuelle_det_nom_bits"],
                       "determinants_emis": ex["determinants_emis"],
                       "masse_par_determinant": ex["masse_par_determinant"],
                       "cond_det_vers_nom": ex["cond_det_vers_nom"],
                       "repartition_par_nom": ex["repartition_par_nom"]})
        # Et les poids, pour ne plus jamais avoir a repondre "je n'ai que les lignes".
        torch.save(politique.state_dict(),
                   os.path.join(dossier_poids, f"politique_b{args.beta}_g{graine}.pt"))
        print(f"{graine:>7} {ex['masse_valide_pct']:>9.2f} {ex['modes_effectifs']:>7.1f} "
              f"{sg:>7.1f} {pl:>7.1f} {branche:>8}")

    compte = Counter(l["branche"] for l in lignes)
    effondres = [l for l in lignes if l["branche"] in ("sg", "pl")]
    n = len(effondres)
    k_sg = sum(1 for l in effondres if l["branche"] == "sg")
    print()
    print(f"  branches : {dict(compte)}")
    print(f"  parmi les {n} runs effondres sur une seule famille : "
          f"{k_sg} singulier, {n - k_sg} pluriel")
    if n:
        print(f"  proportion singulier : {k_sg/n:.3f}  "
              f"Wilson 95 % {wilson(k_sg, n)}")
        print(f"  binomial bilateral contre 1/2 : p = {binomial_bilateral(k_sg, n):.4f}")
        print(f"  binomial bilateral contre 2/3 : p = {binomial_bilateral(k_sg, n, 2/3):.4f}")
    modes = [l["modes_effectifs"] for l in effondres]
    for br in ("sg", "pl"):
        m = [l["modes_effectifs"] for l in effondres if l["branche"] == br]
        if m:
            print(f"  modes effectifs, branche {br} : moyenne {np.mean(m):.2f} "
                  f"+/- {np.std(m):.2f}  (min {min(m)}, max {max(m)}, n = {len(m)})")

    marque = "" if args.chemin == "float32" else f"_{args.chemin}"
    chemin = os.path.join(
        DOSSIER_SORTIE,
        f"balayage_70_graines_b{args.beta}{marque}_{args.debut:03d}_{fin:03d}.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump({"detail": lignes, "n_effondres": n, "k_singulier": k_sg,
                   "wilson95": wilson(k_sg, n) if n else None,
                   "p_contre_moitie": binomial_bilateral(k_sg, n) if n else None},
                  f, indent=2, ensure_ascii=False)
    print(f"\nEcrit dans {chemin}")


if __name__ == "__main__":
    main()
