"""Sonde d'ordre 1 : ou pointe le gradient au tout premier pas, position par position.

Sonde proposee par dipankarsarkar en commentaire de l'article, generalisee ici a
TOUTES les positions au lieu de la seule position 0.

A politique uniforme, le gradient REINFORCE ne voit que les marginales d'ordre 1
E[R | x_p = t] : les termes d'ordre superieur sont non correles avec le token
courant et s'annulent en esperance. Ces marginales se calculent en forme close,
avant tout entrainement.

Ce que la sonde revele et que la position 0 seule ne montre pas : les positions ne
sont pas d'accord entre elles. Le determinant tire vers le pluriel, le nom tire
vers le singulier, et c'est le nom qui porte les traits.

Partie B : la question posee par dipankarsarkar a la fin de son commentaire, et a
laquelle je n'avais pas la reponse. La grammaire longue s'effondre-t-elle au
pluriel elle aussi ? Elle s'effondre bien sur une seule famille, mais c'est le
singulier 4 graines sur 5.
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch

from grammaire import Grammaire
from rl_grammaire import (PolitiqueGRU, fixer_graine, entrainer, DOSSIER_SORTIE)


def marginales_ordre1(g):
    """E[R | x_p = t] exact, toutes positions, politique uniforme sur les autres.

    Forme close : la recompense est une moyenne de sous-scores dont chacun ne
    depend que d'une position (structure) ou d'une paire (accords). Aucun
    echantillonnage, aucune enumeration des V^L sequences.
    """
    V, L = g.taille, g.longueur
    i_nom = g.positions["nom"]
    # Part de structure attendue a politique uniforme, avant conditionnement
    struct_base = sum(len(g.tokens_par_categorie[g.structure[i]]) for i in range(L)) / (V * L)

    marges = {}
    for p in range(L):
        cat_attendue = g.structure[p]
        for t in g.tokens:
            # La position p passe de sa moyenne uniforme a sa valeur conditionnee
            correct = 1.0 if g.traits(t)["categorie"] == cat_attendue else 0.0
            moyen = len(g.tokens_par_categorie[cat_attendue]) / V
            sous_scores = [struct_base + (correct - moyen) / L]

            for categorie, _, traits in g.accords:
                i_cat = g.positions[categorie]
                gauche = [t] if p == i_cat else g.tokens
                droite = [t] if p == i_nom else g.tokens
                total, n = 0.0, 0
                for a in gauche:
                    for b in droite:
                        n += 1
                        if (g.traits(a)["categorie"] != categorie
                                or g.traits(b)["categorie"] != "nom"):
                            continue
                        total += sum(1 for nt in traits
                                     if g._compatible(g.traits(a)[nt], g.traits(b)[nt])) / len(traits)
                sous_scores.append(total / n)

            marges[(p, t)] = sum(sous_scores) / len(sous_scores)
    return marges


def rapport_marginales(g):
    marges = marginales_ordre1(g)
    lignes = []
    for p in range(g.longueur):
        cat = g.structure[p]
        toks = sorted(g.tokens_par_categorie[cat], key=lambda t: -marges[(p, t)])
        lignes.append(f"  pos {p} ({cat:5s}) : "
                      + ", ".join(f"{t} {marges[(p, t)]:.4f}" for t in toks))

    glouton = [max(g.tokens, key=lambda t: marges[(p, t)]) for p in range(g.longueur)]
    ids = [g.index[t] for t in glouton]

    familles = {}
    for cat in ("det", "nom"):
        i = g.positions[cat]
        for nb in ("sg", "pl"):
            vals = [marges[(i, t)] for t in g.tokens_par_categorie[cat]
                    if g.traits(t)["nombre"] == nb]
            familles[f"{cat}_{nb}"] = float(np.mean(vals))

    return {
        "lignes": lignes,
        "glouton": " ".join(glouton),
        "glouton_valide": bool(g.analyser(ids)["valide"]),
        "glouton_reward": round(g.recompense_graduee(ids), 4),
        "familles": {k: round(v, 4) for k, v in familles.items()},
        "ecart_det": round(familles["det_pl"] - familles["det_sg"], 4),
        "ecart_nom": round(familles["nom_sg"] - familles["nom_pl"], 4),
    }


def mesurer_branche(politique, g, n=40000, device="cpu"):
    """Ou la politique entrainee met-elle sa masse : singulier ou pluriel ?

    La grammaire longue a 28,6 M sequences : pas d'enumeration exacte possible,
    on echantillonne. Avec n = 40 000 l'erreur type sur une proportion est
    inferieure a 0,3 point, largement suffisant pour trancher une branche.
    """
    lots = []
    reste = n
    with torch.no_grad():
        while reste > 0:
            taille = min(reste, 10000)
            actions, _, _, _ = politique.generer(g.longueur, taille_lot=taille, device=device)
            lots.append(actions.cpu())
            reste -= taille
    actions = torch.cat(lots).tolist()

    i_det, i_nom = g.positions["det"], g.positions["nom"]
    valides = [s for s in actions if g.analyser(s)["valide"]]

    def repartition(seqs, position, categorie):
        c = Counter()
        for s in seqs:
            tok = g.tokens[s[position]]
            if g.traits(tok)["categorie"] == categorie:
                c[g.traits(tok)["nombre"]] += 1
        total = sum(c.values())
        return {k: round(100 * v / total, 1) for k, v in c.items()} if total else {}, total

    nom_tout, n_nom = repartition(actions, i_nom, "nom")
    det_tout, n_det = repartition(actions, i_det, "det")
    nom_val, _ = repartition(valides, i_nom, "nom")

    return {
        "n": n,
        "validite_pct": round(100 * len(valides) / n, 2),
        "phrases_valides_distinctes": len(set(map(tuple, valides))),
        "nombre_du_nom_toutes_phrases": nom_tout,
        "nombre_du_det_toutes_phrases": det_tout,
        "nombre_du_nom_phrases_valides": nom_val,
        "masse_position_nom_est_un_nom_pct": round(100 * n_nom / n, 1),
        "masse_position_det_est_un_det_pct": round(100 * n_det / n, 1),
    }


def main():
    p = argparse.ArgumentParser(description="Sonde d'ordre 1 et branche de la grammaire longue")
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--graines", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--coef-entropie", type=float, default=0.08)
    p.add_argument("--device", default="cpu")
    p.add_argument("--sans-entrainement", action="store_true",
                   help="partie A seulement")
    args = p.parse_args()

    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    rapport = {"hyperparametres": vars(args), "marginales": {}, "branches_longue": []}

    print("=" * 78)
    print("PARTIE A — marginales d'ordre 1, exactes, avant tout entrainement")
    print("=" * 78)
    for longue in (False, True):
        g = Grammaire(longue=longue)
        r = rapport_marginales(g)
        nom = "longue" if longue else "courte"
        rapport["marginales"][nom] = r
        print(f"\nGrammaire {nom} ({g.taille} tokens, {g.longueur} positions, "
              f"{g.compter_phrases_valides()} phrases valides)")
        for ligne in r["lignes"]:
            print(ligne)
        print(f"  sequence gloutonne d'ordre 1 : '{r['glouton']}' "
              f"| valide = {r['glouton_valide']} | R = {r['glouton_reward']}")
        print(f"  E[R | det pluriel]  {r['familles']['det_pl']:.4f}  vs  "
              f"det singulier {r['familles']['det_sg']:.4f}   "
              f"-> avantage PLURIEL  +{r['ecart_det']:.4f}")
        print(f"  E[R | nom singulier] {r['familles']['nom_sg']:.4f}  vs  "
              f"nom pluriel  {r['familles']['nom_pl']:.4f}   "
              f"-> avantage SINGULIER +{r['ecart_nom']:.4f}")
        print("  les deux positions pointent dans des directions opposees, "
              "d'ou une sequence gloutonne invalide")

    if args.sans_entrainement:
        print("\n(partie B non lancee)")
        return

    print()
    print("=" * 78)
    print("PARTIE B — la grammaire longue s'effondre-t-elle au pluriel ?")
    print("=" * 78)
    longue = Grammaire(longue=True)
    for graine in args.graines:
        fixer_graine(graine)
        politique = PolitiqueGRU(longue.taille).to(args.device)
        res = entrainer(politique, longue, max_episodes=args.episodes,
                        type_recompense="graduee", coef_entropie=args.coef_entropie,
                        verbeux=False, device=args.device, etiquette=f"longue_g{graine}")
        mes = mesurer_branche(politique, longue, device=args.device)
        mes["graine"] = graine
        mes["duree_s"] = res["duree_s"]
        rapport["branches_longue"].append(mes)
        print(f"  graine {graine} ({res['duree_s']:.0f}s) : "
              f"validite {mes['validite_pct']:5.2f} % | "
              f"{mes['phrases_valides_distinctes']:3d} phrases valides distinctes | "
              f"nom : {mes['nombre_du_nom_toutes_phrases']} | "
              f"det : {mes['nombre_du_det_toutes_phrases']}")

    chemin = os.path.join(DOSSIER_SORTIE, "sonde_ordre1.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)
    print(f"\nRapport ecrit dans {chemin}")


if __name__ == "__main__":
    main()
