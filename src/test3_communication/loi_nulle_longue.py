"""
RDTRL — Test 3 : la loi nulle de la concentration, en long, et l'appariement.

Deux questions posees par Dipankar Sarkar apres lecture de docs/TEST3.md :

  1. le maximum de la loi nulle a 20 000 tirages (0,3305) est-il un estimateur de
     quoi que ce soit ? Le seuil « ~0,35 » de §6.1 est bati dessus ;
  2. `concentration()` prend le max sur les attributs colonne par colonne,
     independamment. Un meme attribut peut donc gagner deux positions. Cette
     statistique lit-elle bien ce que §6.1 lui fait dire ?

Ce script mesure les deux. Il calcule sur chaque tirage :

  - `concentration_max`      : la statistique publiee, somme des max par colonne ;
  - `concentration_appariee` : la meme somme sous contrainte d'appariement, une
                               position par attribut (Hongrois exact, 3x3 = 6
                               permutations, donc enumerees et non approchees) ;
  - `double_compte`          : l'argmax par colonne repete-t-il un attribut ?

IDENTITE UTILISEE POUR VECTORISER. Pour un code BIJECTIF, les deux marges de la
loi jointe (attribut i, token en position j) sont exactement uniformes : chaque
valeur d'attribut couvre 9 referents sur 27, et le code atteignant les 27
messages, chaque token couvre 9 positions sur 27. Donc

    I(A_i ; M_j) = somme_{a,t} (c/27) * log2(c/3)      avec c = comptage entier

et la matrice entiere ne depend que des neuf entiers c, ce qui la rend calculable
par table de correspondance. La validation contre `grammaire3.concentration`
n'est donc pas une formalite : elle verifie cette identite.
"""

import argparse
import json
import math
import os
import time
from itertools import permutations

import numpy as np

from grammaire3 import (ATTRIBUT, DOSSIER_SORTIE, INFORMATION_TOTALE, N,
                        N_ATTRIBUTS, N_POSITIONS, N_TOKENS, N_VALEURS,
                        codes_compositionnels, concentration)
from grammaire3 import TOKEN as TOKEN_PAR_POSITION

# g[c] = (c/27) * log2(c/3), le terme d'un comptage entier c dans I(A_i ; M_j).
TERME = np.array([0.0] + [(c / N) * math.log2(c / N_VALEURS)
                          for c in range(1, N // N_VALEURS + 1)])
APPARIEMENTS = list(permutations(range(N_ATTRIBUTS)))


def matrices_information(codes, verifier_bijectivite=True):
    """I(A_i ; M_j) pour un lot de codes BIJECTIFS. Sortie (B, attributs, positions).

    DANGER, et c'est pour ca que la verification est active par defaut. La table
    TERME suppose les deux marges uniformes, ce qui n'est vrai QUE pour un code
    bijectif. Sur un code non bijectif ce chemin rend des nombres faux sans lever
    la moindre erreur.

    Ce n'est pas une precaution theorique : §6.7 mesure que la montee de gradient
    exacte, partie du babil, se pose sur des codes ou 1 a 4 referents entrent en
    collision (E[R] = 23/27 a 26/27). Les codes emergents ne seront donc pas
    bijectifs, et il faudra leur appliquer `grammaire3.matrice_information`, qui
    recalcule les marges au lieu de les supposer.

    `verifier_bijectivite=False` uniquement quand la bijectivite est garantie par
    construction, comme dans le tirage de la loi nulle.
    """
    if verifier_bijectivite:
        for code in codes:
            if len(np.unique(code)) != N:
                raise ValueError(
                    "code non bijectif : ce chemin vectorise suppose les marges "
                    "uniformes. Utiliser grammaire3.matrice_information.")
    lot = codes.shape[0]
    matrices = np.empty((lot, N_ATTRIBUTS, N_POSITIONS))
    # tokens[j, b, r] = token en position j du message emis pour le referent r
    tokens = np.stack([TOKEN_PAR_POSITION[j][codes] for j in range(N_POSITIONS)])
    decalage = (np.arange(lot) * N_VALEURS * N_TOKENS)[:, None]
    for i in range(N_ATTRIBUTS):
        for j in range(N_POSITIONS):
            cases = ATTRIBUT[i][None, :] * N_TOKENS + tokens[j]
            comptes = np.bincount((cases + decalage).ravel(),
                                  minlength=lot * N_VALEURS * N_TOKENS)
            matrices[:, i, j] = TERME[comptes].reshape(lot, -1).sum(axis=1)
    return matrices


def matrices_information_generale(codes):
    """I(A_i ; M_j) pour un lot de codes QUELCONQUES, bijectifs ou non.

    Necessaire depuis §6.5 : les codes atteints ont 1 a 4 collisions, donc la
    table TERME ci-dessus ne s'applique pas. Ici on recalcule la marge des tokens
    au lieu de la supposer uniforme.

    La marge des attributs, elle, reste exactement 9/27 quoi qu'il arrive, chaque
    valeur d'attribut couvrant neuf referents par construction du monde. D'ou

        I(A_i ; M_j) = somme_{a,t} (c_at/27) log2( 3 c_at / n_t )

    avec n_t le nombre de referents dont le message porte le token t en position j.
    Plus lent que le chemin bijectif, mais valide partout.
    """
    lot = codes.shape[0]
    matrices = np.empty((lot, N_ATTRIBUTS, N_POSITIONS))
    tokens = np.stack([TOKEN_PAR_POSITION[j][codes] for j in range(N_POSITIONS)])
    decalage = (np.arange(lot) * N_VALEURS * N_TOKENS)[:, None]
    for i in range(N_ATTRIBUTS):
        for j in range(N_POSITIONS):
            cases = ATTRIBUT[i][None, :] * N_TOKENS + tokens[j]
            comptes = np.bincount((cases + decalage).ravel(),
                                  minlength=lot * N_VALEURS * N_TOKENS)
            comptes = comptes.reshape(lot, N_VALEURS, N_TOKENS).astype(float)
            marge_t = comptes.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                terme = (comptes / N) * np.log2(N_VALEURS * comptes / marge_t)
            matrices[:, i, j] = np.where(comptes > 0, terme, 0.0).sum(axis=(1, 2))
    return matrices


def statistiques(matrices):
    """Les trois quantites, a partir des matrices d'information."""
    par_colonne = matrices.max(axis=1)                     # (B, positions)
    conc_max = par_colonne.sum(axis=1) / INFORMATION_TOTALE

    apparie = np.stack([sum(matrices[:, sigma[j], j] for j in range(N_POSITIONS))
                        for sigma in APPARIEMENTS])
    conc_apparie = apparie.max(axis=0) / INFORMATION_TOTALE

    gagnants = matrices.argmax(axis=1)                     # (B, positions)
    distincts = np.ones(matrices.shape[0], dtype=bool)
    for a in range(N_POSITIONS):
        for b in range(a + 1, N_POSITIONS):
            distincts &= gagnants[:, a] != gagnants[:, b]
    return conc_max, conc_apparie, ~distincts


def valider(n_codes, graine):
    """Le vectorise contre la reference scalaire de grammaire3, code par code."""
    generateur = np.random.default_rng(graine)
    codes = np.array([generateur.permutation(N) for _ in range(n_codes)])
    conc_max, conc_apparie, _ = statistiques(matrices_information(codes))
    reference = np.array([concentration(c) for c in codes])
    ecart = np.abs(conc_max - reference).max()
    # Les codes compositionnels doivent valoir 1 sous LES DEUX statistiques.
    comp = np.array(codes_compositionnels())
    cm, ca, dc = statistiques(matrices_information(comp))
    return {
        "n_codes_verifies": n_codes,
        "ecart_max_au_scalaire": float(ecart),
        "compositionnels_max": [float(cm.min()), float(cm.max())],
        "compositionnels_apparie": [float(ca.min()), float(ca.max())],
        "compositionnels_en_double_compte": int(dc.sum()),
        "appariee_toujours_inferieure": bool((ca <= cm + 1e-12).all()),
    }


SEUILS = (0.28, 0.30, 0.32, 0.35, 0.40, 0.45)
# Seuils d'inflation : le meme traitement par comptes exacts que SEUILS, que le
# script d'origine ne donnait qu'a la concentration. 0,05927 est le maximum
# observe sur les 210 runs, 0,1443 le pire cas de la recherche adverse.
SEUILS_INFLATION = (0.05, 0.05927, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11,
                    0.12, 0.13, 0.1443)


def tirer(n_total, graine, taille_lot, garder_extremes=200_000, reservoir=2_000_000):
    """La loi nulle en long.

    La queue n'est PAS echantillonnee : on maintient exactement les
    `garder_extremes` plus hautes valeurs de tout le tirage, ce qui rend les
    quantiles au-dela de 1 - garder_extremes/n_total exacts et non estimes sur un
    sous-echantillon. Le corps de la loi passe par un reservoir borne, ou une
    estimation suffit largement.
    """
    generateur = np.random.default_rng(graine)
    somme = somme_carres = 0.0
    n_double = 0
    depassements = {s: 0 for s in SEUILS}
    # l'inflation se suit sur TOUT le tirage, pas dans le reservoir : celui-ci
    # cesse de se remplir a `reservoir` et son maximum est alors celui du debut
    # du tirage. Corrige le 16/08/2026, voir §7.31 du carnet.
    infl_max = 0.0
    infl_somme = 0.0
    infl_n_positifs = 0
    infl_somme_positifs = 0.0
    infl_depassements = {s: 0 for s in SEUILS_INFLATION}
    pool_max = np.empty(0)
    pool_apparie = np.empty(0)
    res_max, res_apparie, res_double = [], [], []
    n_reservoir = 0
    pas = 0
    debut = time.time()
    while pas < n_total:
        lot = min(taille_lot, n_total - pas)
        # argsort d'un vecteur de reels rend une permutation : bijectif par
        # construction, donc la verification serait une perte seche a 10^7 tirages
        codes = np.argsort(generateur.random((lot, N)), axis=1)
        cm, ca, dc = statistiques(matrices_information(codes, verifier_bijectivite=False))
        somme += cm.sum()
        somme_carres += (cm * cm).sum()
        n_double += int(dc.sum())
        for s in SEUILS:
            depassements[s] += int((cm >= s).sum())
        inflation = cm - ca
        infl_max = max(infl_max, float(inflation.max()))
        infl_somme += float(inflation.sum())
        positifs = inflation[inflation > 0]
        infl_n_positifs += int(positifs.size)
        infl_somme_positifs += float(positifs.sum())
        for s in SEUILS_INFLATION:
            infl_depassements[s] += int((inflation >= s).sum())
        # queue exacte : on fusionne puis on retaille
        pool_max = np.concatenate([pool_max, cm])
        pool_apparie = np.concatenate([pool_apparie, ca])
        if pool_max.size > garder_extremes:
            indices = np.argpartition(pool_max, -garder_extremes)[-garder_extremes:]
            pool_max, pool_apparie = pool_max[indices], pool_apparie[indices]
        if n_reservoir < reservoir:
            res_max.append(cm)
            res_apparie.append(ca)
            res_double.append(dc)
            n_reservoir += lot
        pas += lot
        if pas % (100 * taille_lot) == 0:
            ecoule = time.time() - debut
            print(f"    {pas:>12,} tirages  {ecoule:6.1f} s  "
                  f"({pas / ecoule / 1000:.0f} k/s)".replace(",", " "), flush=True)
    ordre = np.argsort(pool_max)
    return {
        "n": n_total,
        "moyenne": somme / n_total,
        "ecart_type": math.sqrt(somme_carres / n_total - (somme / n_total) ** 2),
        "n_double_compte": n_double,
        "taux_double_compte": n_double / n_total,
        "depassements": depassements,
        # `taux_double_compte` compte les collisions d'argmax ; `taux_inflation`
        # compte celles qui coutent quelque chose. Elles different de 7 points :
        # une collision peut laisser l'appariement egaler le max exactement.
        "inflation_max": infl_max,
        "inflation_moyenne": infl_somme / n_total,
        "taux_inflation": infl_n_positifs / n_total,
        "inflation_moyenne_si_positive": infl_somme_positifs / infl_n_positifs,
        "inflation_depassements": infl_depassements,
        "hauts_max": pool_max[ordre],
        "hauts_apparie": pool_apparie[ordre],
        "echantillon_max": np.concatenate(res_max),
        "echantillon_apparie": np.concatenate(res_apparie),
        "echantillon_double": np.concatenate(res_double),
        "secondes": time.time() - debut,
    }


def quantile_exact(res, q, appariee=False):
    """Quantile de queue lu dans le pool exact, pas dans le reservoir.

    Le pool est retenu sur la statistique max. Comme l'appariee lui est
    inferieure partout, tout tirage d'appariee superieure a la coupure du pool y
    figure : les quantiles apparies au-dessus de cette coupure sont donc exacts
    eux aussi. On refuse de repondre en dessous plutot que de deviner.
    """
    rang = int(round((1 - q) * res["n"]))
    if rang < 1 or rang > res["hauts_max"].size:
        return None
    if not appariee:
        return float(res["hauts_max"][-rang])
    trie = np.sort(res["hauts_apparie"])
    valeur = float(trie[-rang])
    return valeur if valeur >= res["hauts_max"].min() else None


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="RDTRL — test 3, loi nulle longue")
    parseur.add_argument("--echantillons", type=int, default=10_000_000)
    parseur.add_argument("--graine", type=int, default=0)
    parseur.add_argument("--lot", type=int, default=100_000)
    parseur.add_argument("--validation", type=int, default=3000)
    args = parseur.parse_args()
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    print("=" * 78)
    print("TEST 3 — LOI NULLE LONGUE, ET LA STATISTIQUE APPARIEE")
    print("=" * 78)

    print("\nVALIDATION du chemin vectorise")
    controle = valider(args.validation, args.graine + 9999)
    print(f"  ecart max au calcul scalaire, sur {args.validation} codes : "
          f"{controle['ecart_max_au_scalaire']:.3e}")
    print(f"  les 1296 codes compositionnels, version max      : "
          f"[{controle['compositionnels_max'][0]:.12f}, "
          f"{controle['compositionnels_max'][1]:.12f}]")
    print(f"  les 1296 codes compositionnels, version appariee : "
          f"[{controle['compositionnels_apparie'][0]:.12f}, "
          f"{controle['compositionnels_apparie'][1]:.12f}]")
    print(f"  dont en double compte : {controle['compositionnels_en_double_compte']}")
    print(f"  appariee <= max partout : {controle['appariee_toujours_inferieure']}")
    assert controle["ecart_max_au_scalaire"] < 1e-12, "le vectorise ne reproduit pas"
    assert controle["compositionnels_en_double_compte"] == 0

    print(f"\nTIRAGE — {args.echantillons:,} bijections uniformes, graine {args.graine}"
          .replace(",", " "))
    res = tirer(args.echantillons, args.graine, args.lot)
    print(f"  termine en {res['secondes']:.1f} s")

    ech_max = res["echantillon_max"]
    ech_app = res["echantillon_apparie"]
    corps = (0.001, 0.01, 0.05, 0.5, 0.95, 0.99)
    queue = (0.999, 0.9999, 0.99999, 0.999999)

    print("\n" + "-" * 78)
    print("LA STATISTIQUE PUBLIEE (max par colonne)")
    print("-" * 78)
    print(f"  moyenne    : {res['moyenne']:.4f}   (20 000 tirages : 0,1273)")
    print(f"  ecart-type : {res['ecart_type']:.4f}   (20 000 tirages : 0,0332)")
    print(f"  maximum    : {res['hauts_max'].max():.4f}   (20 000 tirages : 0,3305)")
    for q in corps:
        print(f"    q{100 * q:<9.6g} % : {np.quantile(ech_max, q):.4f}"
              f"   (reservoir de {len(ech_max):,})".replace(",", " "))
    for q in queue:
        v = quantile_exact(res, q)
        print(f"    q{100 * q:<9.6g} % : " + ("hors pool" if v is None else f"{v:.4f}")
              + "   (queue exacte)")
    print()
    for seuil in SEUILS:
        n = res["depassements"][seuil]
        p = n / res["n"]
        # au moins un depassement sur 50 et sur 100 graines, sous la nulle
        print(f"  P(C >= {seuil:.2f}) = {p:.3e}  ({n} sur {res['n']:,})"
              f"   au moins un en 100 tirages : {1 - (1 - p) ** 100:.2e}"
              .replace(",", " "))

    print("\n" + "-" * 78)
    print("LE DOUBLE COMPTE")
    print("-" * 78)
    print(f"  taux global : {100 * res['taux_double_compte']:.1f} % "
          f"(hasard si les argmax etaient uniformes : {100 * (1 - 6 / 27):.1f} %)")
    ordre = np.argsort(ech_max)
    for k, nom in ((200, "les 200 plus hauts"), (2000, "les 2 000 plus hauts")):
        idx = ordre[-k:]
        print(f"  {nom:<22} du reservoir : double compte "
              f"{100 * res['echantillon_double'][idx].mean():.1f} %  "
              f"max {ech_max[idx].mean():.4f}  appariee {ech_app[idx].mean():.4f}  "
              f"inflation {(ech_max[idx] - ech_app[idx]).mean():.4f}")
    inflation = ech_max - ech_app
    print(f"  inflation moyenne, toute la loi : {inflation.mean():.4f}")
    print(f"  inflation maximale observee     : {inflation.max():.4f}")
    print()
    print("  inflation par niveau de concentration :")
    bornes = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 1.01]
    for bas, haut in zip([0.0] + bornes[:-1], bornes):
        dans = (ech_max >= bas) & (ech_max < haut)
        if dans.sum() < 50:
            continue
        print(f"    C dans [{bas:.2f}, {haut:.2f}[  n={int(dans.sum()):>9,}  "
              f"double compte {100 * res['echantillon_double'][dans].mean():5.1f} %  "
              f"inflation moyenne {inflation[dans].mean():.4f}".replace(",", " "))

    print("\n" + "-" * 78)
    print("LA STATISTIQUE APPARIEE (une position par attribut)")
    print("-" * 78)
    print(f"  moyenne    : {ech_app.mean():.4f}")
    print(f"  ecart-type : {ech_app.std():.4f}")
    print(f"  maximum    : {res['hauts_apparie'].max():.4f}")
    for q in corps:
        print(f"    q{100 * q:<9.6g} % : {np.quantile(ech_app, q):.4f}")
    for q in queue:
        v = quantile_exact(res, q, appariee=True)
        print(f"    q{100 * q:<9.6g} % : " + ("hors pool" if v is None else f"{v:.4f}")
              + "   (queue exacte)")

    print("\n" + "-" * 78)
    print("PUISSANCE DU TEST DE §6.2 — deplacement de moyenne detectable")
    print("-" * 78)
    print("  unilateral, p < 0,001, puissance 80 % : delta = 3,93 * sd / racine(n)")
    for nom, ecart in (("max      ", ech_max.std()), ("appariee ", ech_app.std())):
        ligne = f"  {nom} sd {ecart:.4f} :"
        for n_graines in (50, 100):
            ligne += f"   n={n_graines} -> {3.93 * ecart / math.sqrt(n_graines):.4f}"
        print(ligne)

    rapport = {
        "n_echantillons": args.echantillons,
        "graine": args.graine,
        "validation": controle,
        "max": {
            "moyenne": float(res["moyenne"]),
            "ecart_type": float(res["ecart_type"]),
            "maximum": float(res["hauts_max"].max()),
            "quantiles_corps": {str(q): float(np.quantile(ech_max, q)) for q in corps},
            "quantiles_queue_exacts": {str(q): quantile_exact(res, q) for q in queue},
            "depassements_exacts": {str(s): int(n) for s, n in res["depassements"].items()},
            "n_reservoir": int(len(ech_max)),
        },
        "appariee": {
            "moyenne": float(ech_app.mean()),
            "ecart_type": float(ech_app.std()),
            "maximum": float(res["hauts_apparie"].max()),
            "quantiles_corps": {str(q): float(np.quantile(ech_app, q)) for q in corps},
            "quantiles_queue_exacts": {str(q): quantile_exact(res, q, appariee=True)
                                       for q in queue},
        },
        "double_compte": {
            "taux_global": float(res["taux_double_compte"]),
            "taux_200_plus_hauts": float(res["echantillon_double"][ordre[-200:]].mean()),
            "inflation_moyenne_globale": float(inflation.mean()),
            "inflation_moyenne_200_plus_hauts": float(
                (ech_max[ordre[-200:]] - ech_app[ordre[-200:]]).mean()),
            "inflation_maximale": float(inflation.max()),
        },
        "secondes": res["secondes"],
    }
    nom = f"loi_nulle_longue_n{args.echantillons}_g{args.graine}"
    with open(os.path.join(DOSSIER_SORTIE, nom + ".json"), "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(DOSSIER_SORTIE, nom + ".npz"),
                        hauts_max=res["hauts_max"], hauts_apparie=res["hauts_apparie"],
                        echantillon_max=ech_max, echantillon_apparie=ech_app)
    print(f"\nEcrit dans {DOSSIER_SORTIE} sous {nom}")
