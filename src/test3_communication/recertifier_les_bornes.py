"""RDTRL — sa regle appliquee aux fichiers ou personne n'a regarde.

Sa regle du dix-neuvieme tour : une valeur produite par une recherche se rapporte
avec la fraction de l'espace contre laquelle elle a ete certifiee. Je l'ai adoptee
au tour vingt. Le meme jour, j'ai publie `realisabilite_treillis.py`, dont les
deux montees bougent par TRANSPOSITIONS SEULES — 351 mouvements sur 3276, soit
10,7 % du voisinage — et dont la partie A conclut a une NON-REALISABILITE.

Or §7.37 a etabli le fait qui rend ce choix le pire possible : sur 85 arrets
faussement declares optimaux, **85 s'echappaient par 3-cycle et 0 par
transposition**. La moitie exhaustive du voisinage ne peut pas echouer ; c'est
l'autre qui porte tout. `realisabilite_treillis.py` n'utilise que celle qui ne
trouve rien.

Deuxieme chose que personne n'a imprimee nulle part : ces montees s'arretent soit
faute de voisin meilleur, soit parce que le budget de pas est epuise. Le premier
cas est un optimum local, le second une troncature. Aucun des deux fichiers ne
distingue. Compte ici, par bras.

Enumeration des defauts numeriques qui bornent une recherche dans tout le depot,
et etat de chacun, en fin de fichier.
"""

import numpy as np

from grammaire3 import N
from realisabilite_treillis import (PAIRES, inflation, matrices_information,
                                    voisins)

TRIPLES = [(i, j, k) for i in range(N) for j in range(i + 1, N)
           for k in range(j + 1, N)]


def voisins_3cycle_complet(code):
    v = np.repeat(code[None, :], len(TRIPLES), axis=0)
    for n, (i, j, k) in enumerate(TRIPLES):
        v[n, i], v[n, j], v[n, k] = code[j], code[k], code[i]
    return v


def voisinage_complet(code):
    return np.concatenate([voisins(code), voisins_3cycle_complet(code)])


def monter_suivi(code, objectif, pas=300):
    """Copie de `realisabilite_treillis.monter`, avec la RAISON de l'arret."""
    valeur = float(objectif(code[None, :])[0])
    for etape in range(pas):
        cand = voisins(code)
        vals = objectif(cand)
        k = int(vals.argmax())
        if vals[k] <= valeur + 1e-12:
            return valeur, code, "pas de voisin meilleur", etape
        code, valeur = cand[k].copy(), float(vals[k])
    return valeur, code, "budget epuise", pas


def certifier(code, valeur, objectif):
    """Un voisin du voisinage COMPLET fait-il mieux ?"""
    cand = voisinage_complet(code)
    vals = objectif(cand)
    k = int(vals.argmax())
    return float(vals[k]) - valeur, len(cand)


def poursuivre(code, valeur, objectif, pas=300):
    """Certifier un pas ne certifie pas la borne. On termine sous les 3276."""
    for _ in range(pas):
        cand = voisinage_complet(code)
        vals = objectif(cand)
        k = int(vals.argmax())
        if vals[k] <= valeur + 1e-12:
            return valeur, code, False
        code, valeur = cand[k].copy(), float(vals[k])
    return valeur, code, True


def campagne(nom, objectif, n_departs, graine):
    generateur = np.random.default_rng(graine)
    tronques = 0
    faux_optima = 0
    gains = []
    par_type = {"transposition": 0, "3cycle": 0}
    meilleur, meilleur_apres = -np.inf, -np.inf
    optima_avant, optima_apres = set(), set()
    tronques_complet = 0
    for _ in range(n_departs):
        v, c, raison, etape = monter_suivi(generateur.permutation(N), objectif)
        tronques += (raison == "budget epuise")
        gain, taille = certifier(c, v, objectif)
        meilleur = max(meilleur, v)
        optima_avant.add(round(v, 10))
        if gain > 1e-12:
            faux_optima += 1
            gains.append(gain)
            cand = voisinage_complet(c)
            vals = objectif(cand)
            k = int(vals.argmax())
            par_type["transposition" if k < len(PAIRES) else "3cycle"] += 1
        v2, _, coupe = poursuivre(c, v, objectif)
        tronques_complet += coupe
        meilleur_apres = max(meilleur_apres, v2)
        optima_apres.add(round(v2, 10))
    gains = np.array(gains) if gains else np.array([0.0])
    print(f"\n  {nom}")
    print(f"    departs {n_departs}   voisinage utilise {len(PAIRES)} / "
          f"{len(PAIRES) + len(TRIPLES)}  ({len(PAIRES) / (len(PAIRES) + len(TRIPLES)):.1%})")
    print(f"    arrets par budget epuise            {tronques:>5} / {n_departs}")
    print(f"    arrets qui ne sont PAS des optima   {faux_optima:>5} / {n_departs}"
          f"   ({faux_optima / n_departs:.1%})")
    print(f"    echappement par transposition       {par_type['transposition']:>5}"
          f"   par 3-cycle {par_type['3cycle']:>5}")
    print(f"    gain d'echappement  min {gains.min():.12f}  median "
          f"{np.median(gains):.12f}  max {gains.max():.12f}")
    print(f"    optima distincts  {len(optima_avant):>4}  ->  apres poursuite "
          f"sous les 3276  {len(optima_apres):>4}"
          f"   (budget epuise {tronques_complet})")
    print(f"    maximum publie {meilleur:.12f}  ->  apres poursuite complete "
          f"{meilleur_apres:.12f}   ecart {meilleur_apres - meilleur:+.12f}")
    return meilleur, meilleur_apres


if __name__ == "__main__":
    print("=== RE-CERTIFICATION DE `realisabilite_treillis.py` ===")
    print("  Le fichier que j'ai ecrit POUR LUI au tour vingt, dans le message ou")
    print("  j'adoptais sa regle. Ses deux montees bougent par transpositions seules.")

    for attribut in range(3):
        def objectif_a(lot, a=attribut):
            M = matrices_information(lot, verifier_bijectivite=False)
            return M[:, a, :].min(axis=1)
        campagne(f"A. attribut {attribut} : max du min_j I(A_{attribut} ; M_j)",
                 objectif_a, 400, 4242)

    print("\n  cible de realisabilite du sommet du treillis : 0,521362144")

    campagne("B. inflation, les 600 montees du certificat hors ligne",
             inflation, 600, 2026)
