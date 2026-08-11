"""
RDTRL — Test 3, §6.2 : la dynamique tire-t-elle vraiment au hasard parmi les codes ?

§6.1 a repondu a 20 graines. §6.2 est le test lui-meme, et sa raison d'etre est la
PUISSANCE, que le document annonce a 50-100 graines. A 20 graines, le deplacement
de moyenne detectable vaut 3,93 sd/racine(20) = 0,029 : une pression faible qui
souleverait tous les runs de 0,02 — le scenario exact que decrivait Dipankar
Sarkar — serait MANQUEE. Tourner a 20 graines et conclure « indiscernable » aurait
donc ete une conclusion que le dispositif ne portait pas.

CE QUE CE SCRIPT AJOUTE A §6.1, ET RIEN D'AUTRE :

  1. les 100 graines annoncees, sur les deux parametrisations equivariantes ;
  2. un resultat NEGATIF enonce comme il doit l'etre, c'est-a-dire avec la borne
     superieure de ce qui aurait ete detecte, et non par « on n'a rien vu » ;
  3. un balayage en beta, parce qu'une conclusion tiree a un seul beta n'est pas
     une propriete du systeme — la region [0,037 ; 0,170] etant bistable (§6.7),
     rien ne dit que ce qui vaut a 0,02 vaut partout dans le regime de code ;
  4. le test de forme, pas seulement de moyenne : sous le theoreme de §6.1 les
     centiles des runs dans leur propre nulle sont uniformes sur [0, 1], ce qui
     se teste par Kolmogorov-Smirnov et pas seulement par un z moyen.

Le point 4 est le plus severe des quatre : une dynamique pourrait tres bien avoir
la bonne moyenne et la mauvaise loi.
"""

import argparse
import json
import math
import os
from collections import Counter

import numpy as np
import torch

from code_emergent import (distance_a_compositionnel, nulle_du_profil, profil,
                           situer)
from grammaire3 import DOSSIER_SORTIE, N
from loi_nulle_longue import matrices_information_generale, statistiques
from representable_atteignable_stable import (EmetteurFactorise,
                                              EmetteurStructure,
                                              EmetteurTabulaire, Recepteur,
                                              lire_code, monter)

torch.set_num_threads(int(os.environ.get("RDTRL_THREADS", "1")))


def kolmogorov_smirnov(centiles):
    """D et p contre l'uniforme sur [0, 1]. Formule asymptotique de Kolmogorov."""
    x = np.sort(np.asarray(centiles, dtype=float))
    n = len(x)
    i = np.arange(1, n + 1)
    d = float(max(np.max(i / n - x), np.max(x - (i - 1) / n)))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam)
                for k in range(1, 101))
    return d, min(max(p, 0.0), 1.0)


def une_serie(classe, n_graines, beta, pas, generateur, cache, nulle_n):
    """n_graines montees exactes, chacune situee dans la nulle de SON profil."""
    runs = []
    for graine in range(n_graines):
        emetteur, recepteur = classe(generateur), Recepteur(generateur)
        recompense = monter(emetteur, recepteur, beta, pas)
        code = lire_code(emetteur)
        p = profil(code)
        cm, ca, _ = statistiques(
            matrices_information_generale(np.asarray(code)[None, :]))
        nulle = nulle_du_profil(p, nulle_n, generateur, cache)
        runs.append({"graine": graine, "beta": beta, "reward": recompense,
                     "collisions": N - len(p), "bijectif": len(p) == N,
                     "concentration_appariee": float(ca[0]),
                     "concentration_max": float(cm[0]),
                     "distance_compositionnel": distance_a_compositionnel(code),
                     "situation": situer(float(ca[0]), nulle["appariee"])})
    return runs


def resumer(runs, etiquette):
    z = np.array([r["situation"]["z"] for r in runs])
    centiles = np.array([r["situation"]["centile"] for r in runs])
    conc = np.array([r["concentration_appariee"] for r in runs])
    nulle_sd = float(np.mean([r["situation"]["nulle_sd"] for r in runs]))
    n = len(runs)
    erreur = float(z.std(ddof=1) / math.sqrt(n))
    d, p = kolmogorov_smirnov(centiles)
    # borne superieure de ce qui aurait ete detecte : bilateral, p<0,05, 80 % de
    # puissance, soit delta = (1,96 + 0,84) sd / racine(n)
    detectable = 2.80 * nulle_sd / math.sqrt(n)
    return {"etiquette": etiquette, "n": n,
            "concentration_moyenne": float(conc.mean()),
            "concentration_sd": float(conc.std(ddof=1)),
            "z_moyen": float(z.mean()), "z_sd": float(z.std(ddof=1)),
            "z_erreur_type": erreur,
            "z_ic95": [float(z.mean() - 1.96 * erreur), float(z.mean() + 1.96 * erreur)],
            "ks_D": d, "ks_p": p,
            "nulle_sd": nulle_sd,
            "deplacement_detectable": float(detectable),
            "runs_au_dela_q999": int((centiles > 0.999).sum()),
            "reward_moyen": float(np.mean([r["reward"] for r in runs])),
            "collisions_moyennes": float(np.mean([r["collisions"] for r in runs])),
            "bijections": int(sum(r["bijectif"] for r in runs)),
            "distance_moyenne": float(np.mean([r["distance_compositionnel"]
                                               for r in runs]))}


def ligne(r):
    return (f"  {r['etiquette']:>22}  {r['n']:>4}  "
            f"{r['concentration_moyenne']:.4f}  "
            f"{r['z_moyen']:+6.2f} ± {r['z_erreur_type']:.2f}  "
            f"[{r['z_ic95'][0]:+.2f}, {r['z_ic95'][1]:+.2f}]  "
            f"{r['ks_p']:6.3f}  {r['runs_au_dela_q999']:>3}/{r['n']:<4}")


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="RDTRL — test 3, §6.2")
    parseur.add_argument("--graines", type=int, default=100)
    parseur.add_argument("--pas", type=int, default=3000)
    parseur.add_argument("--nulle", type=int, default=20000)
    parseur.add_argument("--beta", type=float, default=0.02)
    parseur.add_argument("--balayage", type=float, nargs="*",
                         default=[0.005, 0.01, 0.03, 0.037])
    parseur.add_argument("--balayage-graines", type=int, default=20)
    parseur.add_argument("--graine", type=int, default=0)
    args = parseur.parse_args()
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    generateur = np.random.default_rng(args.graine)
    cache = {}
    rapport = {"graines": args.graines, "beta": args.beta, "pas": args.pas,
               "nulle": args.nulle, "graine": args.graine}

    print("=" * 78)
    print("TEST 3 §6.2 — LA DYNAMIQUE TIRE-T-ELLE AU HASARD PARMI LES CODES ?")
    print("=" * 78)
    print(f"\n  {args.graines} graines, beta = {args.beta}, nulle appariee au profil.")
    print("  Prediction de §6.1, qui est un theoreme pour les parametrisations")
    print("  equivariantes : z = 0, et centiles uniformes sur [0, 1].\n")
    print(f"  {'parametrisation':>22}  {'n':>4}  {'C app':>6}  "
          f"{'z moyen':>13}  {'IC 95 %':>16}  {'KS p':>6}  {'>q99,9':>8}")

    principaux = {}
    for classe in (EmetteurTabulaire, EmetteurFactorise):
        runs = une_serie(classe, args.graines, args.beta, args.pas,
                         generateur, cache, args.nulle)
        principaux[classe.nom] = {"runs": runs, "resume": resumer(runs, classe.nom)}
        print(ligne(principaux[classe.nom]["resume"]), flush=True)
    # temoin : la parametrisation dont §6.1 dit qu'elle brise la symetrie
    runs = une_serie(EmetteurStructure, args.balayage_graines, args.beta,
                     args.pas, generateur, cache, args.nulle)
    principaux["structure"] = {"runs": runs, "resume": resumer(runs, "structure (témoin)")}
    print(ligne(principaux["structure"]["resume"]), flush=True)
    rapport["principaux"] = {k: v["resume"] for k, v in principaux.items()}
    rapport["runs"] = {k: v["runs"] for k, v in principaux.items()}

    print("\n" + "-" * 78)
    print("LE RESULTAT NEGATIF, ENONCE AVEC SA BORNE")
    print("-" * 78)
    print("  « On n'a rien vu » ne veut rien dire sans dire ce qu'on aurait vu.")
    print("  Deplacement de moyenne detectable, bilateral p < 0,05, puissance 80 % :\n")
    for nom in ("tabulaire", "factorise"):
        r = principaux[nom]["resume"]
        print(f"  {nom:>12} : n = {r['n']}, sd de la nulle {r['nulle_sd']:.4f}")
        print(f"               detectable a partir de {r['deplacement_detectable']:.4f} "
              f"de concentration")
        print(f"               observe : {r['concentration_moyenne']:.4f} contre une "
              f"nulle a {r['concentration_moyenne'] - r['z_moyen'] * r['nulle_sd']:.4f}")
        print(f"               => toute selection residuelle est plus petite que "
              f"{r['deplacement_detectable']:.4f}")
    sd = principaux["tabulaire"]["resume"]["nulle_sd"]
    print(f"\n  Deux criteres, et il faut dire lequel — je les avais melanges :")
    print(f"    bilateral p < 0,05, puissance 80 %  (2,80 sd / racine n) :")
    print(f"      a {args.graines} graines {2.80 * sd / math.sqrt(args.graines):.4f}, "
          f"a 20 graines {2.80 * sd / math.sqrt(20):.4f}")
    print(f"    unilateral p < 0,001, puissance 80 % (3,93 sd / racine n), qui est")
    print(f"    celui du tableau de §6.2 :")
    print(f"      a {args.graines} graines {3.93 * sd / math.sqrt(args.graines):.4f}, "
          f"a 20 graines {3.93 * sd / math.sqrt(20):.4f}")
    print(f"  Le scenario « une pression faible souleve tous les runs de 0,02 » est")
    print(f"  donc juste a la limite a 20 graines sous le critere large, et manque")
    print(f"  sous le critere du document. C'est la raison d'etre de ce script.")

    print("\n" + "-" * 78)
    print("BALAYAGE EN BETA — une conclusion a un seul beta n'est pas une propriete")
    print("-" * 78)
    print(f"  {args.balayage_graines} graines par beta, parametrisation tabulaire.\n")
    print(f"  {'beta':>8}  {'E[R]':>7}  {'collisions':>10}  {'C app':>7}  "
          f"{'z moyen':>13}  {'KS p':>6}")
    balayage = []
    for beta in args.balayage:
        runs = une_serie(EmetteurTabulaire, args.balayage_graines, beta, args.pas,
                         generateur, cache, args.nulle)
        r = resumer(runs, f"tabulaire b={beta}")
        r["beta"] = beta
        balayage.append(r)
        print(f"  {beta:8.4f}  {r['reward_moyen']:7.4f}  "
              f"{r['collisions_moyennes']:10.2f}  {r['concentration_moyenne']:7.4f}  "
              f"{r['z_moyen']:+6.2f} ± {r['z_erreur_type']:.2f}  {r['ks_p']:6.3f}",
              flush=True)
    rapport["balayage"] = balayage

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    tous_z = [principaux[n]["resume"]["z_moyen"] for n in ("tabulaire", "factorise")]
    tous_ic = all(abs(principaux[n]["resume"]["z_moyen"])
                  < 1.96 * principaux[n]["resume"]["z_erreur_type"]
                  for n in ("tabulaire", "factorise"))
    tous_ks = all(principaux[n]["resume"]["ks_p"] > 0.05
                  for n in ("tabulaire", "factorise"))
    print(f"  parametrisations equivariantes : z compatible avec 0 -> {tous_ic}")
    print(f"                                   centiles uniformes  -> {tous_ks}")
    print(f"  temoin structure : z = {principaux['structure']['resume']['z_moyen']:+.2f}, "
          f"KS p = {principaux['structure']['resume']['ks_p']:.3f}")
    print("\n  Lu comme il faut : ce n'est pas « la dynamique tire au hasard », c'est")
    print("  « la dynamique tire au hasard SUR L'ORBITE, quand la parametrisation est")
    print("  equivariante ». Le profil de fibres, lui, n'est pas tire au hasard du")
    print("  tout : c'est la dynamique qui le choisit, et c'est pourquoi on conditionne.")

    nom = f"6_2_dynamique_uniforme_{args.graines}graines_b{args.beta}_g{args.graine}.json"
    with open(os.path.join(DOSSIER_SORTIE, nom), "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nEcrit dans {DOSSIER_SORTIE} sous {nom}")
