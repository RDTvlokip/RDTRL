"""Les trois questions restantes, que seule l'enumeration rend posables.

Q-F — PORTRAIT DE PHASE
La politique vit dans un simplexe a 8 000 dimensions, mais sa trajectoire
d'entrainement occupe sans doute un sous-espace minuscule. Une ACP sur les
distributions exactes relevees en cours de route dit combien de composantes
expliquent le mouvement. Si c'est 2 ou 3, toute la dynamique d'entrainement est
un systeme de basse dimension qu'on peut DESSINER. Impossible sans espace
enumerable : il faut les distributions exactes, pas des estimations.

Q-G — DECOMPOSITION DE VARIANCE DE LA RECOMPENSE
La recompense est une fonction sur un cube discret. Sa decomposition ANOVA
separe ce qui est d'ordre 1 (marginal par position, la seule chose que le
gradient voit a politique uniforme) de ce qui est d'ordre 2 (les accords, par
nature des contraintes de paire) et d'ordre 3.

Cela remplace le mot vague "sparse" par une mesure : une recompense n'est pas
dense ou creuse, elle a un SPECTRE. Et la question du piege de curriculum devient
precise : la sequence qui maximise le signal d'ordre 1 est-elle valide, ou le
signal precoce mene-t-il ailleurs que les contraintes tardives ?

Q-H — FONCTIONNELLES CONSERVEES
Plutot que de postuler une quantite conservee, on la cherche. Les fonctionnelles
lineaires constantes le long de la trajectoire sont exactement le complement
orthogonal du sous-espace parcouru : elles tombent de Q-F. Reste a voir si
certaines sont INTERPRETABLES — par exemple la masse totale portee par une
categorie a une position donnee.
"""

import json
import os
from collections import deque
from itertools import product

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grammaire import Grammaire
from rl_grammaire import PolitiqueGRU, distribution_exacte, fixer_graine, DOSSIER_SORTIE

EPISODES = 20000
PERIODE = 250
BETA = 0.02


def trajectoire(grammaire, graine, episodes=EPISODES, beta=BETA, periode=PERIODE):
    """Entraine et releve la distribution EXACTE tous les `periode` episodes."""
    fixer_graine(graine)
    politique = PolitiqueGRU(grammaire.taille)
    optimiseur = torch.optim.Adam(politique.parameters(), lr=1e-3)
    historique = deque(maxlen=100)
    points, episodes_releves = [], []
    _, p0 = distribution_exacte(politique, grammaire)
    points.append(p0.double().numpy())
    episodes_releves.append(0)
    for episode in range(1, episodes + 1):
        actions, log_probs, entropies, _ = politique.generer(grammaire.longueur, taille_lot=1)
        r = grammaire.recompense_graduee(actions[0].tolist())
        baseline = sum(historique) / len(historique) if historique else 0.0
        historique.append(r)
        perte = -(log_probs.sum() * torch.tensor(r - baseline, dtype=torch.float32)) \
                - beta * entropies.sum()
        optimiseur.zero_grad()
        perte.backward()
        torch.nn.utils.clip_grad_norm_(politique.parameters(), 5.0)
        optimiseur.step()
        if episode % periode == 0:
            _, p = distribution_exacte(politique, grammaire)
            points.append(p.double().numpy())
            episodes_releves.append(episode)
    return np.array(points), episodes_releves


def anova(recompenses, taille, longueur):
    """Decomposition ANOVA fonctionnelle sous mesure uniforme sur le cube."""
    R = recompenses.reshape([taille] * longueur)
    mu = R.mean()
    axes = list(range(longueur))
    f1 = []
    for p in axes:
        autres = tuple(a for a in axes if a != p)
        f1.append(R.mean(axis=autres) - mu)
    f2 = {}
    for p in axes:
        for q in axes:
            if q <= p:
                continue
            autres = tuple(a for a in axes if a not in (p, q))
            m = R.mean(axis=autres) if autres else R
            f2[(p, q)] = m - f1[p][:, None] - f1[q][None, :] - mu
    reconstruit = np.full_like(R, mu)
    for p in axes:
        forme = [1] * longueur
        forme[p] = taille
        reconstruit = reconstruit + f1[p].reshape(forme)
    for (p, q), val in f2.items():
        forme = [1] * longueur
        forme[p] = taille
        forme[q] = taille
        reconstruit = reconstruit + val.reshape(forme)
    f3 = R - reconstruit

    v_total = float(R.var())
    v1 = {p: float((f1[p] ** 2).mean()) for p in axes}
    v2 = {k: float((v ** 2).mean()) for k, v in f2.items()}
    v3 = float((f3 ** 2).mean())
    return {"variance_totale": v_total, "ordre1": v1, "ordre2": v2, "ordre3": v3,
            "f1": f1}


if __name__ == "__main__":
    g = Grammaire(longue=False)
    sequences = list(product(range(g.taille), repeat=g.longueur))
    seq_arr = np.array(sequences)
    r_graduee = np.array([g.recompense_graduee(s) for s in sequences])
    r_sparse = np.array([g.recompense_tout_ou_rien(s) for s in sequences])
    valide = np.array([g.analyser(s)["valide"] for s in sequences])
    resultats = {}

    print("=" * 88)
    print("Q-G — SPECTRE DE LA RECOMPENSE : decomposition de variance ANOVA")
    print("=" * 88)
    print("Ce que le gradient voit a politique uniforme = ordre 1 uniquement.")
    print("Les accords sont des contraintes de PAIRE, donc d'ordre 2 par nature.\n")
    for nom, rec in (("GRADUEE", r_graduee), ("TOUT-OU-RIEN", r_sparse)):
        d = anova(rec, g.taille, g.longueur)
        s1 = sum(d["ordre1"].values())
        s2 = sum(d["ordre2"].values())
        total = s1 + s2 + d["ordre3"]
        print(f"  recompense {nom} (variance totale {d['variance_totale']:.6f}) :")
        print(f"    ordre 1 (marginales)   : {s1:.6f}  = {100*s1/total:5.1f} %")
        print(f"    ordre 2 (paires)       : {s2:.6f}  = {100*s2/total:5.1f} %")
        print(f"    ordre 3 (triplet)      : {d['ordre3']:.6f}  = {100*d['ordre3']/total:5.1f} %")
        detail1 = {f"pos{p} ({g.structure[p]})": round(100 * v / total, 1)
                   for p, v in d["ordre1"].items()}
        detail2 = {f"pos{p}-{q}": round(100 * v / total, 1) for (p, q), v in d["ordre2"].items()}
        print(f"    detail ordre 1 : {detail1}")
        print(f"    detail ordre 2 : {detail2}")
        # Le piege de curriculum : la sequence gloutonne d'ordre 1 est-elle valide ?
        gloutonne = tuple(int(np.argmax(d["ordre1_f"] if False else d["f1"][p]))
                          for p in range(g.longueur))
        phrase = " ".join(g.tokens[i] for i in gloutonne)
        analyse = g.analyser(gloutonne)
        print(f"    sequence gloutonne d'ordre 1 : '{phrase}' -> "
              f"valide={analyse['valide']}, R={g.recompense_graduee(gloutonne):.4f}")
        resultats[f"anova_{nom}"] = {
            "ordre1_pct": round(100 * s1 / total, 2),
            "ordre2_pct": round(100 * s2 / total, 2),
            "ordre3_pct": round(100 * d["ordre3"] / total, 2),
            "detail_ordre1": detail1, "detail_ordre2": detail2,
            "gloutonne_ordre1": phrase, "gloutonne_valide": bool(analyse["valide"]),
        }
        print()

    print("=" * 88)
    print("Q-F — PORTRAIT DE PHASE : dimension effective de la trajectoire")
    print("=" * 88)
    trajectoires, releves = [], None
    for graine in (0, 1, 2):
        pts, releves = trajectoire(g, graine)
        trajectoires.append(pts)
        print(f"  graine {graine} : {len(pts)} releves de la distribution exacte")
    toutes = np.concatenate(trajectoires, axis=0)
    centre = toutes.mean(axis=0)
    U, S, Vt = np.linalg.svd(toutes - centre, full_matrices=False)
    variance = S ** 2
    ratio = variance / variance.sum()
    cumul = np.cumsum(ratio)
    print(f"\n  variance expliquee : " +
          ", ".join(f"CP{i+1} {100*ratio[i]:.1f} %" for i in range(6)))
    for seuil in (0.90, 0.99, 0.999):
        k = int(np.searchsorted(cumul, seuil) + 1)
        print(f"  {int(100*seuil)} % du mouvement tient dans {k} dimension(s) "
              f"sur {toutes.shape[1]}")
    resultats["acp"] = {"ratio_6_premieres": [round(float(x), 5) for x in ratio[:6]],
                        "dim_90": int(np.searchsorted(cumul, 0.90) + 1),
                        "dim_99": int(np.searchsorted(cumul, 0.99) + 1),
                        "dim_999": int(np.searchsorted(cumul, 0.999) + 1)}

    # Portrait de phase dans le plan des deux premieres composantes
    ideal = np.zeros(len(sequences))
    ideal[valide] = 1.0 / valide.sum()
    proj_ideal = (ideal - centre) @ Vt[:2].T
    plt.figure(figsize=(9, 7))
    couleurs = ["#1f77b4", "#d62728", "#2ca02c"]
    for k, pts in enumerate(trajectoires):
        proj = (pts - centre) @ Vt[:2].T
        plt.plot(proj[:, 0], proj[:, 1], "-", color=couleurs[k], alpha=0.7,
                 linewidth=1.2, label=f"graine {k}")
        plt.scatter(proj[0, 0], proj[0, 1], color=couleurs[k], marker="o", s=70,
                    edgecolor="black", zorder=3)
        plt.scatter(proj[-1, 0], proj[-1, 1], color=couleurs[k], marker="s", s=70,
                    edgecolor="black", zorder=3)
    plt.scatter(*proj_ideal, color="gold", marker="*", s=400, edgecolor="black",
                zorder=4, label="uniforme sur les 48 (optimum)")
    plt.xlabel(f"CP1 ({100*ratio[0]:.1f} % du mouvement)")
    plt.ylabel(f"CP2 ({100*ratio[1]:.1f} %)")
    plt.title("Portrait de phase de l'entrainement — rond = depart, carre = arrivee")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    chemin = os.path.join(DOSSIER_SORTIE, "portrait_de_phase.png")
    plt.savefig(chemin, dpi=130)
    plt.close()
    print(f"  portrait de phase : {chemin}")
    print()

    print("=" * 88)
    print("Q-H — FONCTIONNELLES APPROXIMATIVEMENT CONSERVEES")
    print("=" * 88)
    print("Amplitude de variation, le long de la trajectoire, de la masse portee")
    print("par chaque categorie a chaque position.\n")
    variations = []
    for position, categorie_attendue in enumerate(g.structure):
        for categorie in sorted(g.tokens_par_categorie):
            indices = [i for i, s in enumerate(sequences)
                       if g.traits(g.tokens[s[position]])["categorie"] == categorie]
            valeurs = np.array([[pts[t, indices].sum() for t in range(len(pts))]
                                for pts in trajectoires])
            amplitude = float(valeurs.max() - valeurs.min())
            variations.append({"position": position, "categorie": categorie,
                               "attendue": categorie == categorie_attendue,
                               "amplitude": round(amplitude, 5),
                               "depart": round(float(valeurs[:, 0].mean()), 4),
                               "arrivee": round(float(valeurs[:, -1].mean()), 4)})
    variations.sort(key=lambda v: v["amplitude"])
    print(f"    {'pos':>4} {'categorie':>10} {'attendue':>9} {'depart':>8} "
          f"{'arrivee':>8} {'amplitude':>10}")
    for v in variations:
        print(f"    {v['position']:>4} {v['categorie']:>10} "
              f"{str(v['attendue']):>9} {v['depart']:>8.4f} {v['arrivee']:>8.4f} "
              f"{v['amplitude']:>10.5f}")
    quasi = [v for v in variations if v["amplitude"] < 0.02]
    print(f"\n  {len(quasi)} fonctionnelle(s) varient de moins de 0.02 sur tout "
          f"l'entrainement :")
    for v in quasi:
        print(f"    position {v['position']} / categorie {v['categorie']} "
              f"(amplitude {v['amplitude']:.5f})")
    resultats["fonctionnelles"] = variations

    with open(os.path.join(DOSSIER_SORTIE, "trajectoire_structure.json"), "w",
              encoding="utf-8") as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)
    print(f"\nEcrit dans {DOSSIER_SORTIE}")
