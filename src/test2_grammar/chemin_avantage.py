"""Trois lignes d'avantage, trois chemins numeriques, et ce que ca change.

Trouve par dipankarsarkar. Les boucles d'entrainement du depot sont
algebriquement identiques a taille_lot = 1, mais la ligne qui calcule l'avantage
n'existe pas en une seule version :

  rl_grammaire.py:141            (recompenses_t - baseline).detach()
  stabilite_et_trajectoire.py:79 torch.tensor(r - baseline, dtype=torch.float32)
  parametrisation_et_recuit.py:90  idem
  localisation_effondrement.py:55  idem
  trajectoire_couplage.py:84     torch.tensor(r - base).detach()

La premiere soustrait EN float32 : recompenses_t est deja float32 et baseline est
un flottant Python, donc la promotion tenseur-scalaire arrondit la baseline avant
la soustraction. Les quatre autres soustraient deux float64 puis arrondissent.

Correction a sa lecture : la derniere n'est PAS un troisieme chemin. Il supposait
qu'omettre dtype laisse l'avantage en float64 et promeut la perte. Faux en torch :
torch.get_default_dtype() vaut float32, donc torch.tensor(x) sur un flottant
Python rend un tenseur float32. Verifie : meme dtype, meme valeur au bit pres,
meme dtype de perte. Il y a DEUX chemins, pas trois, et trajectoire_couplage est
du cote des trois autres.

Ce n'est pas une subtilite sans effet : recompense_graduee rend des tiers et des
neuviemes, dont aucun n'est exact en binaire, donc les deux arrondis divergent
des les premiers pas. Et un seul bit suffit a changer une trajectoire, parce que
distribution.sample() est un seuil sur un tirage uniforme : il finit par faire
basculer un token, apres quoi les deux runs ne partagent plus que la graine.

PARTIE A - de combien les trois lignes different, sur le flux de recompenses reel.
PARTIE B - les trois chemins entraines, sondes exactement, meme graine.
           Question ouverte a laquelle personne n'a repondu : le chemin float32
           culmine-t-il lui aussi a 24 modes en cours de route ? De ca depend si
           l'ecart d'arret precoce du carnet vaut +12,5 modes ou +5,4.
"""

import argparse
import json
import os
from collections import deque

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

from grammaire import Grammaire
from rl_grammaire import PolitiqueGRU, fixer_graine, DOSSIER_SORTIE
from optimum_produit import contexte
from trajectoire_couplage import sonder

# Les trois chemins, isoles. Seule cette fonction differe entre les boucles.
CHEMINS = {
    "float32": lambda r, b: (torch.tensor([r], dtype=torch.float32) - b).detach(),
    "float64_arrondi": lambda r, b: torch.tensor(r - b, dtype=torch.float32),
    "float64": lambda r, b: torch.tensor(r - b).detach(),
}


def flux_recompenses(g, graine, etapes):
    """Enregistre (recompense, baseline) pas a pas sur un run reel."""
    fixer_graine(graine)
    politique = PolitiqueGRU(g.taille)
    opt = torch.optim.Adam(politique.parameters(), lr=1e-3)
    memoire, flux = deque(maxlen=100), []
    for _ in range(etapes):
        actions, log_probs, entropies, _ = politique.generer(g.longueur, taille_lot=1)
        r = g.recompense_graduee(actions[0].tolist())
        b = sum(memoire) / len(memoire) if memoire else 0.0
        flux.append((r, b))
        memoire.append(r)
        avantage = CHEMINS["float32"](r, b)
        perte = -(log_probs.sum() * avantage) - 0.02 * entropies.sum()
        opt.zero_grad()
        perte.backward()
        torch.nn.utils.clip_grad_norm_(politique.parameters(), 5.0)
        opt.step()
    return flux


def partie_a(g, graines, etapes):
    print("=" * 80)
    print("PARTIE A - de combien les trois lignes different sur le flux reel")
    print("=" * 80)
    resume = []
    for graine in graines:
        flux = flux_recompenses(g, graine, etapes)
        a = np.array([float(CHEMINS["float32"](r, b)) for r, b in flux])
        c = np.array([float(CHEMINS["float64_arrondi"](r, b)) for r, b in flux])
        d = np.array([float(CHEMINS["float64"](r, b)) for r, b in flux])
        diff = a != c
        premier = int(np.argmax(diff)) + 1 if diff.any() else None
        # Fenetre precoce contre fenetre saturee : la recompense sature vers 1,0
        recompenses = np.array([r for r, _ in flux])
        tot = np.abs(c)
        rel = np.where(tot > 0, np.abs(a - c) / np.maximum(tot, 1e-30), 0.0)
        n_tot = min(2000, len(flux))
        tard = slice(n_tot, len(flux)) if len(flux) > n_tot else slice(0, 0)
        ligne = {
            "graine": graine,
            "premier_desaccord": premier,
            "pct_desaccord_2000_premiers": round(100 * float(diff[:n_tot].mean()), 1),
            "pct_desaccord_apres_2000": (round(100 * float(diff[tard].mean()), 1)
                                         if diff[tard].size else None),
            "ecart_relatif_max": float(rel.max()),
            "float64_vs_arrondi_identiques": bool(np.array_equal(c, d)),
            "recompense_moyenne_fin": round(float(recompenses[-100:].mean()), 4),
        }
        resume.append(ligne)
        print(f"  graine {graine} : premier desaccord au pas {premier} | "
              f"{ligne['pct_desaccord_2000_premiers']} % des 2 000 premiers pas | "
              f"{ligne['pct_desaccord_apres_2000']} % ensuite | "
              f"ecart relatif max {ligne['ecart_relatif_max']:.2e}")
    print(f"\n  float64 et float64_arrondi sont la MEME chose "
          f"({resume[0]['float64_vs_arrondi_identiques']}), valeur ET dtype.")
    print(f"    torch.get_default_dtype() vaut float32, donc torch.tensor(x) sur un")
    print(f"    flottant Python rend un tenseur float32 et non float64. Omettre dtype")
    print(f"    ne promeut rien : c'est vrai en numpy, faux en torch.")
    print(f"    => il y a DEUX chemins numeriques dans le depot, pas trois.")
    return resume


def entrainer_trace(g, sequences, recompenses_t, valide, chemin, beta, graine,
                    etapes, periode):
    """Meme boucle pour les trois, seule la ligne d'avantage change."""
    fixer_graine(graine)
    politique = PolitiqueGRU(g.taille)
    opt = torch.optim.Adam(politique.parameters(), lr=1e-3)
    memoire = deque(maxlen=100)
    calcul = CHEMINS[chemin]
    h = [{"pas": 0, **sonder(politique, sequences, recompenses_t, valide, g)}]
    for pas in range(1, etapes + 1):
        actions, log_probs, entropies, _ = politique.generer(g.longueur, taille_lot=1)
        r = g.recompense_graduee(actions[0].tolist())
        b = sum(memoire) / len(memoire) if memoire else 0.0
        memoire.append(r)
        perte = -(log_probs.sum() * calcul(r, b)) - beta * entropies.sum()
        opt.zero_grad()
        perte.backward()
        torch.nn.utils.clip_grad_norm_(politique.parameters(), 5.0)
        opt.step()
        if pas % periode == 0:
            h.append({"pas": pas, **sonder(politique, sequences, recompenses_t, valide, g)})
    return h


def partie_b(g, sequences, recompenses_t, valide, graines, beta, etapes, periode,
             chemins=None):
    print()
    print("=" * 80)
    print("PARTIE B - les trois chemins entraines, sondes exactement")
    print("=" * 80)
    print(f"{'chemin':>16} {'graine':>7} {'modes max':>10} {'au pas':>8} "
          f"{'modes fin':>10} {'I fin':>8} {'valide% fin':>12} {'branche':>8}")
    lignes = []
    for chemin in (chemins or list(CHEMINS)):
        for graine in graines:
            h = entrainer_trace(g, sequences, recompenses_t, valide, chemin,
                                beta, graine, etapes, periode)
            mo = [e["modes_effectifs"] for e in h]
            k = int(np.argmax(mo))
            fin = h[-1]
            # Branche : on relit la masse par determinant de la sonde finale
            masse = fin["masse_determinants"]
            pl = sum(v for d, v in masse.items() if g.traits(d)["nombre"] == "pl")
            branche = "pl" if pl > 0.5 else "sg"
            ligne = {"chemin": chemin, "graine": graine,
                     "modes_max": mo[k], "pas_du_max": h[k]["pas"],
                     "modes_fin": fin["modes_effectifs"],
                     "I_fin": fin["information_mutuelle_det_nom_bits"],
                     "valide_fin_pct": fin["masse_valide_pct"],
                     "branche": branche,
                     "masse_determinants_fin": masse,
                     "historique": h}
            lignes.append(ligne)
            print(f"{chemin:>16} {graine:>7} {mo[k]:>10.2f} {h[k]['pas']:>8} "
                  f"{fin['modes_effectifs']:>10.2f} "
                  f"{fin['information_mutuelle_det_nom_bits']:>8.4f} "
                  f"{fin['masse_valide_pct']:>12.2f} {branche:>8}")
    return lignes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--beta", type=float, default=0.02)
    p.add_argument("--graines", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--etapes", type=int, default=20000)
    p.add_argument("--periode", type=int, default=250)
    p.add_argument("--partie", choices=["a", "b", "ab"], default="ab")
    p.add_argument("--chemins", nargs="+", choices=list(CHEMINS), default=None,
                   help="sous-ensemble de chemins, pour paralleliser par processus")
    args = p.parse_args()

    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    g = Grammaire(longue=False)
    sequences, recompenses_t, valide = contexte(g)
    rapport = {"beta": args.beta, "etapes": args.etapes}

    if args.partie in ("a", "ab"):
        rapport["partie_a"] = partie_a(g, args.graines, args.etapes)
    if args.partie in ("b", "ab"):
        rapport["partie_b"] = partie_b(g, sequences, recompenses_t, valide,
                                       args.graines, args.beta, args.etapes,
                                       args.periode, args.chemins)
        print()
        print("Lecture : si le chemin float32 culmine lui aussi vers 24 modes, l'ecart")
        print("d'arret precoce annonce au carnet (24,0 max contre 11,5 final, soit")
        print("+12,5) est celui d'un seul chemin et vaut +5,4 sur l'autre.")

    suffixe = "_".join(args.chemins) if args.chemins else "tous"
    chemin = os.path.join(DOSSIER_SORTIE, f"chemin_avantage_{suffixe}.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)
    print(f"\nEcrit dans {chemin}")


if __name__ == "__main__":
    main()
