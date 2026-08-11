"""
RDTRL — Test 3 : le max par colonne, ou l'appariement ? Trois epreuves.

`concentration()` prend, colonne par colonne, l'attribut le mieux explique, sans
contrainte entre colonnes. Un meme attribut peut donc gagner deux positions.
Dipankar Sarkar a montre que ca arrive dans 74,6 % des tirages uniformes, et
demande si le max tient lieu d'un appariement dont je n'avais pas encore eu besoin.

Ca ne se tranche pas par gout. §6.1 lit le scalaire comme une POSITION dans
l'espace des codes parfaits — pas comme un binaire. Une position se juge contre
quelque chose d'exterieur a la statistique. Trois epreuves, de la plus decisive a
la plus indirecte, parce qu'aucune ne suffit seule :

  A. STRUCTURE CONNUE PAR CONSTRUCTION. On fabrique des codes ou exactement k
     positions sur 3 encodent proprement un attribut, le reste etant brouille
     conditionnellement. k est la verite terrain, il n'est ni estime ni approche.
     Un bon lecteur de structure doit rendre k/3 et rien de plus.

  B. DISTANCE COMBINATOIRE. d(code) = min sur les 1 296 codes compositionnels du
     nombre de referents envoyes ailleurs. Les 1 296 contenant deja tous les
     renommages, d ne punit pas un code d'etre compositionnel dans un autre
     etiquetage. C'est une metrique GROSSIERE — elle punit durement un code
     compositionnel sur deux attributs et brouille sur le troisieme — donc elle
     ne sert ici qu'a verifier que la conclusion de A ne depend pas du choix de A.

  C. PIRE CAS. Recherche directe du code qui maximise l'ecart entre les deux
     statistiques. Si cet ecart reste minuscule, la question est academique ; s'il
     est grand, il faut savoir jusqu'ou le max peut etre pousse par de la seule
     redondance.
"""

import argparse
import json
import os

import numpy as np

from grammaire3 import (DOSSIER_SORTIE, INDEX_MESSAGE, N, N_ATTRIBUTS,
                        N_POSITIONS, N_TOKENS, N_VALEURS, REFERENTS,
                        codes_compositionnels)
from loi_nulle_longue import matrices_information, statistiques

COMPOSITIONNELS = np.array(codes_compositionnels(), dtype=np.int8)


def code_a_k_positions_propres(k, generateur):
    """Un code ou exactement k positions encodent proprement un attribut.

    Les positions propres portent une bijection valeur -> token. Les positions
    restantes encodent les attributs restants par une permutation tiree au hasard
    CONDITIONNELLEMENT aux valeurs propres, ce qui garantit la bijectivite globale
    tout en detruisant la lisibilite position par position.

    k = 3 redonne exactement un code compositionnel, k = 0 une bijection uniforme.
    """
    sigma = generateur.permutation(N_ATTRIBUTS)      # attribut encode a la position j
    bijections = [generateur.permutation(N_TOKENS) for _ in range(k)]
    n_suffixes = N_TOKENS ** (N_POSITIONS - k)
    melanges = {}
    code = np.empty(N, dtype=np.int64)
    for r, referent in enumerate(REFERENTS):
        cle = tuple(int(referent[sigma[j]]) for j in range(k))
        if cle not in melanges:
            melanges[cle] = generateur.permutation(n_suffixes)
        # les attributs non propres, lus en base 3, puis brouilles DANS la cellule
        reste = 0
        for j in range(k, N_POSITIONS):
            reste = reste * N_VALEURS + int(referent[sigma[j]])
        brouille = int(melanges[cle][reste])
        tokens = [int(bijections[j][referent[sigma[j]]]) for j in range(k)]
        tokens += [0] * (N_POSITIONS - k)
        for j in range(N_POSITIONS - 1, k - 1, -1):
            tokens[j] = brouille % N_TOKENS
            brouille //= N_TOKENS
        code[r] = INDEX_MESSAGE[tuple(tokens)]
    return code


def distance_a_compositionnel(codes, morceau=400):
    """min sur les 1 296 codes compositionnels du nombre de referents differents."""
    sortie = np.empty(codes.shape[0], dtype=np.int16)
    for debut in range(0, codes.shape[0], morceau):
        bout = codes[debut:debut + morceau].astype(np.int8)
        differents = (bout[:, None, :] != COMPOSITIONNELS[None, :, :]).sum(axis=2)
        sortie[debut:debut + morceau] = differents.min(axis=1)
    return sortie


def echelle_transpositions(n_par_niveau, n_transpositions, generateur):
    """Codes obtenus en s'eloignant d'un compositionnel par k transpositions."""
    codes = []
    for k in range(n_transpositions + 1):
        depart = COMPOSITIONNELS[generateur.integers(0, len(COMPOSITIONNELS),
                                                     size=n_par_niveau)]
        lot = depart.astype(np.int64).copy()
        lignes = np.arange(n_par_niveau)
        for _ in range(k):
            i = generateur.integers(0, N, size=n_par_niveau)
            # garantit i != j, sinon la transposition est un non-evenement
            j = (i + 1 + generateur.integers(0, N - 1, size=n_par_niveau)) % N
            vi, vj = lot[lignes, i].copy(), lot[lignes, j].copy()
            lot[lignes, i], lot[lignes, j] = vj, vi
        codes.append(lot)
    return np.concatenate(codes)


def concordance(a, b, verite, generateur, n_paires=2_000_000):
    """Sur des paires departageables par `verite`, taux d'accord de chaque statistique.

    Les deux statistiques sont jugees sur LES MEMES paires : l'ecart entre elles
    ne peut donc pas venir de l'echantillon. Les ex aequo comptent une demi-reussite,
    faute de quoi la statistique la plus discrete serait penalisee pour rien.
    """
    i = generateur.integers(0, len(verite), size=n_paires)
    j = generateur.integers(0, len(verite), size=n_paires)
    utile = verite[i] != verite[j]
    i, j = i[utile], j[utile]
    attendu = verite[i] < verite[j]          # plus proche => concentration plus haute
    sortie = {"n_paires_departageables": int(utile.sum())}
    for nom, v in (("max", a), ("appariee", b)):
        juste = ((v[i] > v[j]) == attendu) & (v[i] != v[j])
        sortie["concordance_" + nom] = float(juste.mean() + 0.5 * (v[i] == v[j]).mean())
    return sortie


def recherche_pire_cas(objectif, generateur, n_restarts=24, n_pas=60):
    """Montee de gradient exacte sur les 351 transpositions, plusieurs departs."""
    paires = [(i, j) for i in range(N) for j in range(i + 1, N)]
    meilleur, meilleur_code = -1.0, None
    for _ in range(n_restarts):
        code = generateur.permutation(N)
        valeur = objectif(code[None, :])[0]
        for _ in range(n_pas):
            voisins = np.repeat(code[None, :], len(paires), axis=0)
            for n, (i, j) in enumerate(paires):
                voisins[n, i], voisins[n, j] = code[j], code[i]
            valeurs = objectif(voisins)
            k = int(valeurs.argmax())
            if valeurs[k] <= valeur + 1e-12:
                break
            code, valeur = voisins[k].copy(), float(valeurs[k])
        if valeur > meilleur:
            meilleur, meilleur_code = valeur, code.copy()
    return meilleur, meilleur_code


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description="RDTRL — max contre appariement")
    parseur.add_argument("--par-famille", type=int, default=4000)
    parseur.add_argument("--par-niveau", type=int, default=3000)
    parseur.add_argument("--transpositions", type=int, default=14)
    parseur.add_argument("--graine", type=int, default=7)
    args = parseur.parse_args()
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    generateur = np.random.default_rng(args.graine)

    print("=" * 78)
    print("TEST 3 — LE MAX PAR COLONNE CONTRE L'APPARIEMENT")
    print("=" * 78)

    print("\n" + "-" * 78)
    print("A. STRUCTURE CONNUE PAR CONSTRUCTION")
    print("-" * 78)
    print("  k positions sur 3 encodent proprement un attribut ; le reste est brouille.")
    print("  Un lecteur honnete de la structure doit rendre k/3.\n")
    print(f"  {'k':>2}  {'attendu':>8}  {'max':>17}  {'appariee':>17}  "
          f"{'double compte':>13}  {'inflation':>9}")
    familles = {}
    for k in range(N_ATTRIBUTS + 1):
        codes = np.array([code_a_k_positions_propres(k, generateur)
                          for _ in range(args.par_famille)])
        assert all(len(set(c.tolist())) == N for c in codes[::391]), "code non bijectif"
        cm, ca, dc = statistiques(matrices_information(codes))
        familles[k] = {"n": int(len(codes)), "attendu": k / N_ATTRIBUTS,
                       "max_moyenne": float(cm.mean()), "max_sd": float(cm.std()),
                       "appariee_moyenne": float(ca.mean()), "appariee_sd": float(ca.std()),
                       "taux_double_compte": float(dc.mean()),
                       "inflation": float((cm - ca).mean()),
                       "biais_max": float(cm.mean() - k / N_ATTRIBUTS),
                       "biais_appariee": float(ca.mean() - k / N_ATTRIBUTS)}
        f = familles[k]
        print(f"  {k:>2}  {f['attendu']:8.4f}  {f['max_moyenne']:9.4f} "
              f"±{f['max_sd']:.4f}  {f['appariee_moyenne']:9.4f} ±{f['appariee_sd']:.4f}  "
              f"{100 * f['taux_double_compte']:12.1f} %  {f['inflation']:9.4f}")
    print("\n  ecart a la valeur attendue (ce qui est credite en trop) :")
    for k in range(N_ATTRIBUTS + 1):
        f = familles[k]
        print(f"    k={k}   max {f['biais_max']:+.4f}      appariee {f['biais_appariee']:+.4f}")

    print("\n" + "-" * 78)
    print("B. DISTANCE COMBINATOIRE AU PLUS PROCHE CODE COMPOSITIONNEL")
    print("-" * 78)
    codes = echelle_transpositions(args.par_niveau, args.transpositions, generateur)
    conc_max, conc_app, double = statistiques(matrices_information(codes))
    d = distance_a_compositionnel(codes)
    print(f"  {len(codes):,} codes, distances de {d.min()} a {d.max()}\n"
          .replace(",", " "))
    print(f"  {'d':>3}  {'n':>6}  {'max':>8}  {'appariee':>9}  {'inflation':>10}  "
          f"{'double compte':>14}")
    table = []
    for valeur in range(0, int(d.max()) + 1):
        dans = d == valeur
        if dans.sum() < 30:
            continue
        ligne = {"d": int(valeur), "n": int(dans.sum()),
                 "max": float(conc_max[dans].mean()),
                 "appariee": float(conc_app[dans].mean()),
                 "inflation": float((conc_max - conc_app)[dans].mean()),
                 "taux_double_compte": float(double[dans].mean())}
        table.append(ligne)
        print(f"  {valeur:>3}  {ligne['n']:>6}  {ligne['max']:8.4f}  "
              f"{ligne['appariee']:9.4f}  {ligne['inflation']:10.4f}  "
              f"{100 * ligne['taux_double_compte']:13.1f} %")

    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])

    rho_max = spearman(conc_max, -d.astype(float))
    rho_app = spearman(conc_app, -d.astype(float))
    conc = concordance(conc_max, conc_app, d, generateur)
    print(f"\n  Spearman avec -d :  max {rho_max:.4f}   appariee {rho_app:.4f}")
    print(f"  sur {conc['n_paires_departageables']:,} paires de distances differentes :"
          .replace(",", " "))
    print(f"    max      classe bien {100 * conc['concordance_max']:.2f} %")
    print(f"    appariee classe bien {100 * conc['concordance_appariee']:.2f} %")
    print(f"    ecart : {100 * (conc['concordance_appariee'] - conc['concordance_max']):+.2f} pt")

    print("\n" + "-" * 78)
    print("C. PIRE CAS — jusqu'ou la redondance peut-elle pousser le max ?")
    print("-" * 78)

    def objectif_inflation(lot):
        cm, ca, _ = statistiques(matrices_information(lot))
        return cm - ca

    def objectif_max_en_double(lot):
        cm, _, dc = statistiques(matrices_information(lot))
        return np.where(dc, cm, -1.0)      # interdit les codes sans double compte

    ensemble_compositionnel = {tuple(int(x) for x in c) for c in COMPOSITIONNELS}

    def objectif_max_non_compositionnel(lot):
        cm, _, _ = statistiques(matrices_information(lot))
        interdit = np.array([tuple(int(x) for x in c) in ensemble_compositionnel
                             for c in lot])
        return np.where(interdit, -1.0, cm)

    inflation_max, code_inflation = recherche_pire_cas(objectif_inflation, generateur)
    cm_i, ca_i, dc_i = statistiques(matrices_information(code_inflation[None, :]))
    print(f"  ecart max - appariee, maximise      : {inflation_max:.4f}")
    print(f"    ce code vaut max {cm_i[0]:.4f} et appariee {ca_i[0]:.4f}")

    plafond, code_plafond = recherche_pire_cas(objectif_max_en_double, generateur)
    cm_p, ca_p, dc_p = statistiques(matrices_information(code_plafond[None, :]))
    print(f"  plus haut max atteint EN double compte : {plafond:.4f}")
    print(f"    ce code vaut appariee {ca_p[0]:.4f}, soit {cm_p[0] - ca_p[0]:.4f} de trop")

    second, code_second = recherche_pire_cas(objectif_max_non_compositionnel, generateur)
    print(f"  plus haute concentration NON compositionnelle : {second:.4f}")
    print(f"    le sommet de l'echelle est donc isole : 1,0000 puis {second:.4f}")
    print(f"  rappel : max de la loi nulle sur 10 000 000 tirages = 0,3979")
    print("  (ces trois valeurs sont des MINORANTS : la montee est locale, pas exhaustive)")

    rapport = {"graine": args.graine,
               "familles_k_positions_propres": familles,
               "par_distance": table,
               "spearman_max": rho_max, "spearman_appariee": rho_app,
               "concordance": conc,
               "pire_cas": {
                   "inflation_maximale": float(inflation_max),
                   "code_inflation_max": [int(x) for x in code_inflation],
                   "max_du_code_inflation": float(cm_i[0]),
                   "appariee_du_code_inflation": float(ca_i[0]),
                   "plafond_max_en_double_compte": float(plafond),
                   "appariee_du_code_plafond": float(ca_p[0]),
                   "code_plafond": [int(x) for x in code_plafond],
                   "max_non_compositionnel": float(second),
                   "code_non_compositionnel": [int(x) for x in code_second],
                   "note": "minorants : montee locale sur transpositions, non exhaustive"}}
    nom = (f"appariement_{args.par_famille}par_famille"
           f"_{args.par_niveau}par_niveau_g{args.graine}.json")
    with open(os.path.join(DOSSIER_SORTIE, nom), "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)
    print(f"\nEcrit dans {DOSSIER_SORTIE} sous {nom}")
