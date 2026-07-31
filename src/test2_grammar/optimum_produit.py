"""Le plafond est-il un mur, ou l'optimum d'un probleme restreint ?

Question 1 de l'approfondissement du 31/07/2026.

Fait a expliquer : sur 37 runs singuliers, 19 sont EXACTEMENT a 12,0 modes et les
autres sur des produits d'entiers. Une borne dit "au plus 12". Les donnees disent
"exactement 12, une fois sur deux". Un plafond atteint aussi precisement n'est pas
une contrainte subie, c'est un optimum.

Hypothese : REINFORCE a beta constant resout EXACTEMENT le probleme restreint aux
politiques sans couplage, et echoue uniquement a quitter cette classe.

Protocole. On optimise le meme objectif E[R] + beta*H, par gradient exact et sans
echantillonnage, sur trois classes emboitees :

  produit   trois lois independantes p(d), p(n), p(v) : I(det;nom) = 0 par
            construction, c'est la classe des politiques sans couplage ;
  libre     la loi jointe complete sur les 8 000 sequences, tabulaire, aucune
            contrainte : c'est l'optimum de Gibbs ;
  reseau    le GRU, deja mesure ailleurs, rappele ici pour comparaison.

Si l'optimum produit tombe sur 12,0 modes a beta = 0,02, l'hypothese tient et le
"plafond" devient un optimum contraint. Sinon elle tombe.
"""

import argparse
import json
import os
from itertools import product as iproduct

import numpy as np
import torch

from grammaire import Grammaire
from rl_grammaire import DOSSIER_SORTIE

ETAPES = 6000
LR = 0.05


def contexte(g):
    """Sequences, recompenses et masque de validite, une fois pour toutes."""
    sequences = torch.tensor(list(iproduct(range(g.taille), repeat=g.longueur)),
                             dtype=torch.long)
    recompenses = torch.tensor([g.recompense_graduee(s.tolist()) for s in sequences],
                               dtype=torch.float64)
    valide = torch.tensor([g.analyser(s.tolist())["valide"] for s in sequences])
    return sequences, recompenses, valide


def mesures(p, sequences, recompenses, valide, g):
    """Masse valide, modes effectifs sur les valides, et I(det ; nom)."""
    p = p / p.sum()
    masse = float(p[valide].sum())
    q = p[valide] / max(masse, 1e-30)
    h = float(-(q * q.clamp_min(1e-30).log2()).sum())

    V = g.taille
    jointe = p.reshape([V] * g.longueur).numpy()
    i_det, i_nom = g.positions["det"], g.positions["nom"]
    axes = tuple(k for k in range(g.longueur) if k not in (i_det, i_nom))
    m = jointe.sum(axis=axes)
    if i_det > i_nom:
        m = m.T
    m = m / m.sum()
    pd, pn = m.sum(1, keepdims=True), m.sum(0, keepdims=True)
    prod = pd * pn
    nz = (m > 1e-14) & (prod > 1e-30)
    im = float((m[nz] * np.log2(m[nz] / prod[nz])).sum())
    return {"masse_valide_pct": round(100 * masse, 3),
            "modes_effectifs": round(2 ** h, 2),
            "information_mutuelle_det_nom_bits": round(im, 4)}


def optimiser_produit(g, sequences, recompenses, beta, graine, etapes=ETAPES, lr=LR):
    """Trois logits independants, un par position. I(det;nom) = 0 par construction."""
    torch.manual_seed(graine)
    logits = [torch.randn(g.taille, dtype=torch.float64, requires_grad=True)
              for _ in range(g.longueur)]
    opt = torch.optim.Adam(logits, lr=lr)
    for _ in range(etapes):
        lps = [torch.log_softmax(l, dim=0) for l in logits]
        log_p = sum(lps[t][sequences[:, t]] for t in range(g.longueur))
        p = log_p.exp()
        objectif = (p * recompenses).sum() + beta * (-(p * log_p).sum())
        opt.zero_grad()
        (-objectif).backward()
        opt.step()
    with torch.no_grad():
        lps = [torch.log_softmax(l, dim=0) for l in logits]
        log_p = sum(lps[t][sequences[:, t]] for t in range(g.longueur))
        return log_p.exp()


def optimiser_libre(g, sequences, recompenses, beta, graine, etapes=ETAPES, lr=LR):
    """Loi jointe tabulaire, aucune contrainte : l'optimum de Gibbs."""
    torch.manual_seed(graine)
    logits = torch.randn(len(sequences), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(etapes):
        log_p = torch.log_softmax(logits, dim=0)
        p = log_p.exp()
        objectif = (p * recompenses).sum() + beta * (-(p * log_p).sum())
        opt.zero_grad()
        (-objectif).backward()
        opt.step()
    with torch.no_grad():
        return torch.log_softmax(logits, dim=0).exp()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.08])
    p.add_argument("--graines", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()

    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    g = Grammaire(longue=False)
    sequences, recompenses, valide = contexte(g)

    # Reference analytique : le plus grand produit entierement valide, tous coins
    # confondus, et coin par coin.
    from produit_et_saturation import plus_grand_produit
    global_max, _ = plus_grand_produit(g, g.tokens_par_categorie["nom"])
    par_coin = {}
    for nb in ("sg", "pl"):
        noms = [n for n in g.tokens_par_categorie["nom"] if g.traits(n)["nombre"] == nb]
        par_coin[nb] = plus_grand_produit(g, noms)[0]

    print("=" * 84)
    print("L'OPTIMUM DE LA CLASSE PRODUIT COINCIDE-T-IL AVEC LE PLAFOND MESURE ?")
    print("=" * 84)
    print(f"  plus grand produit valide : global {global_max} | "
          f"coin sg {par_coin['sg']} | coin pl {par_coin['pl']}")
    print(f"  REINFORCE echantillonne a beta=0.02 : 19 runs sur 37 exactement a 12,0")
    print()
    print(f"{'beta':>6} {'classe':>8} {'graine':>7} {'valide%':>9} {'modes':>8} {'I(d;n)':>9}")

    rapport = {"plus_grand_produit": {"global": global_max, **par_coin}, "runs": []}
    for beta in args.betas:
        for classe, f in (("produit", optimiser_produit), ("libre", optimiser_libre)):
            for graine in args.graines:
                p_opt = f(g, sequences, recompenses, beta, graine)
                m = mesures(p_opt, sequences, recompenses, valide, g)
                m.update({"beta": beta, "classe": classe, "graine": graine})
                rapport["runs"].append(m)
                print(f"{beta:>6} {classe:>8} {graine:>7} {m['masse_valide_pct']:>9.3f} "
                      f"{m['modes_effectifs']:>8.2f} "
                      f"{m['information_mutuelle_det_nom_bits']:>9.4f}")
        print()

    chemin = os.path.join(DOSSIER_SORTIE, "optimum_produit.json")
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)
    print(f"Ecrit dans {chemin}")
    print()
    print("Lecture : si la classe 'produit' se pose sur 12,0 modes la ou REINFORCE")
    print("se pose, alors le plafond n'est pas un mur subi mais l'optimum exact du")
    print("probleme restreint, et le seul echec de REINFORCE est de ne pas quitter")
    print("la classe. Si elle se pose ailleurs, l'hypothese tombe.")


if __name__ == "__main__":
    main()
