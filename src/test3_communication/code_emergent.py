"""
RDTRL — Test 3, §6.1 : quel code emerge, et ou tombe-t-il ?

Precede de la correction annoncee en §6.7 et rappelee en §6.5, et sans laquelle
cette etape ne veut rien dire.

    LA LOI NULLE DE §6.1 EST TIREE SUR DES BIJECTIONS. Les codes atteints n'en
    sont pas : §6.5 mesure 2 a 5 collisions en moyenne, et une seule bijection sur
    vingt. Comparer une concentration emergente a une loi nulle bijective compare
    deux supports differents, et l'ecart va dans le sens du resultat qu'on predit.

Trois sorties possibles etaient ouvertes en §6.5 : ne garder que les runs
bijectifs (1 sur 20, ca vide l'experience), garder la nulle bijective (malhonnete),
ou tirer la nulle sur la classe REELLEMENT ATTEINTE. C'est la troisieme qui est
faite ici, et elle est appariee run par run.

APPARIEMENT. A chaque code emergent on associe son PROFIL DE FIBRES : le multi-
ensemble des tailles de preimages. Un code bijectif a pour profil 27 fois 1 ; un
code a deux collisions peut avoir 25 singletons et une fibre de 3, ou 24
singletons et deux paires, et ces deux la n'ont pas la meme loi nulle. La
reference d'un run est donc tiree uniformement parmi les applications de MEME
profil, et pas parmi les bijections ni meme parmi les applications de meme
nombre de collisions.

Le tirage uniforme dans cette classe : on permute les 27 referents au hasard, on
coupe la permutation en blocs aux tailles du profil, et on donne a chaque bloc un
message distinct tire au hasard. Chaque partition de tailles donnees est atteinte
le meme nombre de fois, donc le tirage est bien uniforme.

POURQUOI CETTE CLASSE-LA ET PAS UNE AUTRE, ET C'EST LE POINT IMPORTANT. Le groupe
S_27 x S_27 agit sur les applications par (pi, rho) . c = pi o c o rho^-1, et deux
applications sont dans la MEME ORBITE si et seulement si elles ont le meme profil
de fibres : rho realigne les fibres, pi realigne les images. La classe de profil
EST donc l'orbite.

Or §6.7 etablit que la parametrisation tabulaire est equivariante sous le
renommage des messages, et §6.5 qu'elle l'est aussi sous celui des referents,
faute de partage entre referents. Elle est donc equivariante sous le groupe
entier, et la loi de sortie CONDITIONNEE AU PROFIL est exactement uniforme sur ce
profil.

La nulle appariee n'est donc pas une reference « plus proche » que la bijective.
C'est la seule qui soit theoriquement correcte, et elle transforme §6.1 en test a
prediction sans aucun parametre libre : z = 0 en tabulaire, par theoreme.

TROIS PREDICTIONS, ECRITES AVANT DE LIRE LES CHIFFRES.

  tabulaire : z = 0. Equivariante des DEUX cotes, donc uniforme sur l'orbite
              entiere. C'est un theoreme, pas une attente ; un ecart signifierait
              que l'implementation brise une symetrie quelque part.

  factorise : je ne sais pas, et c'est la ligne interessante. Elle est
              equivariante cote REFERENT seulement. L'orbite sous rho seul est
              l'ensemble des applications de meme profil ET DE MEME IMAGE : la
              garantie d'uniformite porte sur l'affectation des fibres, pas sur le
              CHOIX DES MESSAGES UTILISES. Or la concentration depend de la
              structure en tokens des messages utilises. Un z non nul est donc
              possible, et il serait entierement imputable a la factorisation
              cote message.

  structure : z nettement positif. Voir §6.5, ou l'ecart brut vaut +0,2950.
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch

from grammaire3 import DOSSIER_SORTIE, N, codes_compositionnels
from loi_nulle_longue import matrices_information_generale, statistiques
from representable_atteignable_stable import (EmetteurFactorise,
                                              EmetteurStructure,
                                              EmetteurTabulaire, Recepteur,
                                              lire_code, monter)

torch.set_num_threads(int(os.environ.get("RDTRL_THREADS", "1")))
COMPOSITIONNELS = np.array(codes_compositionnels(), dtype=np.int8)


def profil(code):
    """Multi-ensemble trie des tailles de preimages. Un code bijectif : 27 fois 1."""
    return tuple(sorted(Counter(np.asarray(code).tolist()).values(), reverse=True))


def tirer_profil(taille_profil, n, generateur):
    """n applications tirees uniformement parmi celles de profil donne."""
    tailles = np.asarray(taille_profil)
    assert tailles.sum() == N, "un profil doit couvrir les 27 referents"
    bloc_de_position = np.repeat(np.arange(len(tailles)), tailles)
    ordre = np.argsort(generateur.random((n, N)), axis=1)
    messages = np.argsort(generateur.random((n, N)), axis=1)[:, :len(tailles)]
    codes = np.empty((n, N), dtype=np.int64)
    lignes = np.arange(n)[:, None]
    codes[lignes, ordre] = messages[:, bloc_de_position]
    return codes


def distance_a_compositionnel(code):
    """min sur les 1 296 codes compositionnels du nombre de referents differents.

    Lecture independante de l'information mutuelle, valable pour une application
    quelconque et pas seulement pour une bijection.
    """
    return int((np.asarray(code, dtype=np.int8)[None, :] != COMPOSITIONNELS).sum(1).min())


def nulle_du_profil(taille_profil, n, generateur, cache):
    if taille_profil not in cache:
        codes = tirer_profil(taille_profil, n, generateur)
        cm, ca, _ = statistiques(matrices_information_generale(codes))
        cache[taille_profil] = {"max": cm, "appariee": ca}
    return cache[taille_profil]


def situer(valeur, echantillon):
    """Position d'une valeur dans sa loi nulle : centile et ecarts-types."""
    return {"centile": float((echantillon < valeur).mean()),
            "z": float((valeur - echantillon.mean()) / echantillon.std()),
            "nulle_moyenne": float(echantillon.mean()),
            "nulle_sd": float(echantillon.std())}


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="RDTRL — test 3, §6.1")
    parseur.add_argument("--graines", type=int, default=20)
    parseur.add_argument("--pas", type=int, default=3000)
    parseur.add_argument("--nulle", type=int, default=20000)
    parseur.add_argument("--beta", type=float, default=0.02)
    parseur.add_argument("--graine", type=int, default=0)
    args = parseur.parse_args()
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    generateur = np.random.default_rng(args.graine)
    classes = (EmetteurTabulaire, EmetteurFactorise, EmetteurStructure)

    print("=" * 78)
    print("TEST 3 §6.1 — QUEL CODE EMERGE, ET OU TOMBE-T-IL ?")
    print("=" * 78)
    print(f"\n  beta = {args.beta}, {args.graines} graines par parametrisation,")
    print(f"  loi nulle appariee au profil de fibres, {args.nulle} tirages par profil.\n")

    cache = {}
    runs = []
    for classe in classes:
        for graine in range(args.graines):
            emetteur, recepteur = classe(generateur), Recepteur(generateur)
            recompense = monter(emetteur, recepteur, args.beta, args.pas)
            code = lire_code(emetteur)
            p = profil(code)
            matrice = matrices_information_generale(np.asarray(code)[None, :])
            cm, ca, _ = statistiques(matrice)
            nulle = nulle_du_profil(p, args.nulle, generateur, cache)
            runs.append({
                "parametrisation": classe.nom, "graine": graine,
                "reward": recompense, "profil": list(p),
                "images": len(p), "collisions": N - len(p),
                "bijectif": len(p) == N,
                "distance_compositionnel": distance_a_compositionnel(code),
                "concentration_max": float(cm[0]),
                "concentration_appariee": float(ca[0]),
                "situation_max": situer(float(cm[0]), nulle["max"]),
                "situation_appariee": situer(float(ca[0]), nulle["appariee"])})
        faits = [r for r in runs if r["parametrisation"] == classe.nom]
        print(f"  {classe.nom:>10} : {len(faits)} runs, "
              f"{len(set(tuple(r['profil']) for r in faits))} profils distincts")

    print("\n" + "-" * 78)
    print("CE QUE LA CORRECTION CHANGE — nulle appariee contre nulle bijective")
    print("-" * 78)
    bijective = nulle_du_profil(tuple([1] * N), args.nulle, generateur, cache)
    print(f"  {'profil':>26}  {'n runs':>6}  "
          f"{'nulle appariee':>22}  {'ecart a la bijective':>21}")
    profils = Counter(tuple(r["profil"]) for r in runs)
    lignes_profil = []
    for p, combien in sorted(profils.items(), key=lambda x: -x[1]):
        nulle = cache[p]
        ecart = nulle["appariee"].mean() - bijective["appariee"].mean()
        resume = f"{len(p)} images, fibres {max(p)}"
        lignes_profil.append({"profil": list(p), "n_runs": combien,
                              "nulle_appariee_moyenne": float(nulle["appariee"].mean()),
                              "nulle_appariee_sd": float(nulle["appariee"].std()),
                              "ecart_a_la_bijective": float(ecart)})
        print(f"  {resume:>26}  {combien:>6}  "
              f"{nulle['appariee'].mean():.4f} ± {nulle['appariee'].std():.4f}  "
              f"{ecart:>+21.4f}")
    print(f"\n  nulle bijective, pour memoire : "
          f"{bijective['appariee'].mean():.4f} ± {bijective['appariee'].std():.4f}")
    print("  Un ecart positif signifie que la nulle bijective SOUS-ESTIME la")
    print("  reference, donc qu'elle exagere tout ecart mesure contre elle.")

    print("\n" + "-" * 78)
    print("§6.1 — OU TOMBE LE CODE EMERGENT, DANS SA PROPRE LOI NULLE")
    print("-" * 78)
    print(f"  {'parametrisation':>16}  {'C appariee':>18}  {'z apparie':>16}  "
          f"{'> q99,9':>8}  {'d au compo':>11}")
    resume = {}
    for classe in classes:
        faits = [r for r in runs if r["parametrisation"] == classe.nom]
        conc = np.array([r["concentration_appariee"] for r in faits])
        z = np.array([r["situation_appariee"]["z"] for r in faits])
        centiles = np.array([r["situation_appariee"]["centile"] for r in faits])
        distances = np.array([r["distance_compositionnel"] for r in faits])
        au_dela = int((centiles > 0.999).sum())
        resume[classe.nom] = {
            "concentration_appariee_moyenne": float(conc.mean()),
            "concentration_appariee_sd": float(conc.std()),
            "z_moyen": float(z.mean()), "z_sd": float(z.std()),
            "runs_au_dela_q999": au_dela, "n": len(faits),
            "distance_compositionnel_moyenne": float(distances.mean()),
            "distance_compositionnel_min": int(distances.min())}
        print(f"  {classe.nom:>16}  {conc.mean():7.4f} ± {conc.std():.4f}  "
              f"{z.mean():+8.2f} ± {z.std():.2f}  {au_dela:>3} / {len(faits):<3}  "
              f"{distances.mean():5.1f} (min {distances.min()})")

    print("\n  Rappel de l'engagement de §5, enregistre le 29/07/2026 : la")
    print("  concentration emergente doit etre statistiquement INDISCERNABLE de")
    print("  celle d'un tirage uniforme. Lu sur la nulle appariee, run par run.")
    for classe in classes:
        r = resume[classe.nom]
        verdict = ("INDISCERNABLE" if abs(r["z_moyen"]) < 2 else
                   "ECART SIGNIFICATIF")
        print(f"    {classe.nom:>16} : z moyen {r['z_moyen']:+.2f}  ->  {verdict}")

    rapport = {"beta": args.beta, "graines": args.graines, "pas": args.pas,
               "nulle": args.nulle, "graine": args.graine,
               "runs": runs, "profils": lignes_profil, "resume": resume,
               "nulle_bijective": {
                   "appariee_moyenne": float(bijective["appariee"].mean()),
                   "appariee_sd": float(bijective["appariee"].std()),
                   "max_moyenne": float(bijective["max"].mean())}}
    nom = f"6_1_code_emergent_b{args.beta}_{args.graines}graines_g{args.graine}.json"
    with open(os.path.join(DOSSIER_SORTIE, nom), "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nEcrit dans {DOSSIER_SORTIE} sous {nom}")
