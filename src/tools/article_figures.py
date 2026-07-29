"""Regenere toutes les figures de l'article, en anglais, depuis les donnees sauvegardees.

Les graphiques produits pendant les experiences ont des labels francais, ce qui
detonne dans un article en anglais. Ce script les reconstruit a partir des CSV,
des JSON et des poids deja sur disque — plus deux figures qui manquaient pour des
resultats majeurs (l'optimum de Gibbs et le recuit).

Sortie : dossier figures/
"""

import csv
import json
import os
from collections import deque
from itertools import product

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(RACINE, "src", "test2_grammar"))

from grammar import Grammaire
from rl_grammar import PolitiqueGRU, analyse_exacte, distribution_exacte, fixer_graine

FIGURES = os.path.join(RACINE, "figures")
R1 = os.path.join(RACINE, "results")
R2 = os.path.join(RACINE, "results_test2")

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25,
                     "font.size": 10})

BLEU, ROUGE, VERT, ORANGE = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"


def lire_csv(chemin, colonne):
    valeurs = []
    with open(chemin, encoding="utf-8") as f:
        for ligne in csv.DictReader(f):
            valeurs.append(float(ligne[colonne]))
    return valeurs


def fig_test1_recompenses():
    """Les trois formes de recompense du test 1 sur un meme axe."""
    plt.figure(figsize=(10, 5))
    donnees = [("per-position reward", "run_principal.csv", BLEU),
               ("Levenshtein reward", "run_levenshtein.csv", VERT),
               ("all-or-nothing reward", "run_sparse.csv", ROUGE)]
    for etiquette, fichier, couleur in donnees:
        y = lire_csv(os.path.join(R1, fichier), "recompense_moyenne_100")
        plt.plot(range(1, len(y) + 1), y, color=couleur, linewidth=1.6, label=etiquette)
    plt.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.6)
    plt.xlabel("episode")
    plt.ylabel("reward (100-episode moving average)")
    plt.ylim(-0.03, 1.05)
    plt.title("Test 1 — copying 'le chat dort' from random weights\n"
              "all-or-nothing stays at exactly zero for 30,000 episodes")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "01_test1_reward_shapes.png"))
    plt.close()


def fig_test1_heatmap():
    """Distribution apprise, position par caractere."""
    chemin = os.path.join(R1, "politique_le_chat_dort.pt")
    if not os.path.exists(chemin):
        print("  (poids du test 1 absents, heatmap ignoree)")
        return
    cible = "le chat dort"
    caracteres = sorted(set(" ") | set(cible) | set("le chien dort"))
    index = {c: i for i, c in enumerate(caracteres)}

    politique = PolitiqueGRU(len(caracteres))
    politique.load_state_dict(torch.load(chemin))
    politique.eval()
    with torch.no_grad():
        _, _, _, probas = politique.generer(len(cible), taille_lot=1, greedy=True)
    matrice = probas[0].numpy().T

    plt.figure(figsize=(11, 5.5))
    plt.imshow(matrice, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="probability (greedy decoding)")
    plt.yticks(range(len(caracteres)),
               [f"'{c}'" if c != " " else "'_'" for c in caracteres])
    plt.xticks(range(len(cible)),
               [f"{i+1}\n'{c}'" if c != " " else f"{i+1}\n'_'" for i, c in enumerate(cible)])
    plt.xlabel("position in the sequence (and target character)")
    plt.ylabel("vocabulary character")
    plt.title("Test 1 — what the policy became: a 12-entry lookup table")
    for pos, c in enumerate(cible):
        plt.gca().add_patch(plt.Rectangle((pos - 0.5, index[c] - 0.5), 1, 1,
                                          fill=False, edgecolor="red", linewidth=1.5))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "02_test1_heatmap.png"))
    plt.close()


def fig_test1_transfert():
    """Transfert contre depart de zero sur la cible perturbee."""
    plt.figure(figsize=(10, 5))
    for etiquette, fichier, couleur in [
            ("transfer (weights kept)", "run_transfert.csv", BLEU),
            ("from scratch (weights reset)", "run_depuis_zero.csv", ROUGE)]:
        y = lire_csv(os.path.join(R1, fichier), "recompense_moyenne_100")
        plt.plot(range(1, len(y) + 1), y, color=couleur, linewidth=1.6, label=etiquette)
    plt.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.6)
    plt.xlabel("episode")
    plt.ylabel("reward (100-episode moving average)")
    plt.ylim(-0.03, 1.05)
    plt.title("Test 1 — transfer to 'le chien dort' (5 of 13 positions shared)\n"
              "×1.74 speedup here — but ×0.91 with zero positional overlap")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "03_test1_transfer.png"))
    plt.close()


def fig_test2_grammaires():
    """Grammaire courte et longue, gradue contre tout-ou-rien."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    configs = [
        (axes[0], "Short grammar — 8,000 sequences, 0.600 % random validity",
         [("graded", "run_courte_ent0.08.csv", BLEU),
          ("all-or-nothing", "run_courte_tout_ou_rien.csv", ROUGE)]),
        (axes[1], "Long grammar — 28.6M sequences, 0.001 % random validity",
         [("graded", "run_longue_graduee.csv", BLEU),
          ("all-or-nothing", "run_longue_tout_ou_rien.csv", ROUGE)]),
    ]
    for ax, titre, series in configs:
        for etiquette, fichier, couleur in series:
            chemin = os.path.join(R2, fichier)
            if not os.path.exists(chemin):
                continue
            y = lire_csv(chemin, "validite_100")
            ax.plot(range(1, len(y) + 1), [100 * v for v in y],
                    color=couleur, linewidth=1.6, label=etiquette)
        ax.set_title(titre, fontsize=10)
        ax.set_xlabel("episode")
        ax.set_ylim(-3, 105)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("% grammatically valid sentences (last 100)")
    fig.suptitle("Test 2 — the same all-or-nothing signal succeeds or fails "
                 "purely on the random hit rate", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "04_test2_grammars.png"))
    plt.close()


def fig_compromis(grammaire):
    """Validite, uniformite, modes selon beta + la frontiere, multi-graines."""
    chemin = os.path.join(R2, "balayage_graines.json")
    if not os.path.exists(chemin):
        print("  (balayage multi-graines absent)")
        return
    with open(chemin, encoding="utf-8") as f:
        donnees = json.load(f)
    detail, synthese = donnees["detail"], donnees["synthese"]
    coefs = [s["coef"] for s in synthese]
    total = grammaire.compter_phrases_valides()

    fig, (g, d) = plt.subplots(1, 2, figsize=(13.5, 5))
    x = range(len(coefs))
    v_moy = [s["valide_moy"] for s in synthese]
    v_ec = [s["valide_ec"] for s in synthese]
    m_moy = [100 * s["modes_moy"] / total for s in synthese]
    m_ec = [100 * s["modes_ec"] / total for s in synthese]
    g.errorbar(x, v_moy, yerr=v_ec, fmt="o-", color=BLEU, capsize=3,
               label="grammatical mass (exact)")
    g.errorbar(x, m_moy, yerr=m_ec, fmt="s-", color=ROUGE, capsize=3,
               label=f"effective modes / {total}")
    g.set_xticks(list(x))
    g.set_xticklabels([str(c) for c in coefs])
    g.set_xlabel("entropy coefficient β")
    g.set_ylabel("%")
    g.set_ylim(-5, 108)
    g.legend(fontsize=9)
    g.set_title("Grammaticality and diversity vs entropy bonus\n"
                "mean ± std over 3 seeds", fontsize=10)

    couleurs = {0: BLEU, 1: ROUGE, 2: VERT}
    for graine in (0, 1, 2):
        pts = [l for l in detail if l["graine"] == graine]
        d.plot([l["modes_effectifs"] for l in pts],
               [l["masse_valide_pct"] for l in pts],
               "o-", color=couleurs[graine], alpha=0.75, markersize=5,
               label=f"seed {graine}")
    d.axvline(total, color="black", linestyle=":", linewidth=1, alpha=0.6)
    d.text(total - 1.5, 50, f"{total} solutions\n(uniform = the optimum)",
           fontsize=8, alpha=0.7, ha="right")
    d.set_xlabel("effective modes 2^H (valid sentences actually used)")
    d.set_ylabel("grammatical mass (%)")
    d.set_xlim(0, total * 1.12)
    d.set_ylim(-5, 108)
    d.legend(fontsize=9)
    d.set_title("The validity × diversity frontier\n"
                "every point is one training run", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "05_test2_tradeoff.png"))
    plt.close()


def fig_gibbs(grammaire):
    """Optimum theorique contre atteint : taxe de mise en forme et ecart."""
    sequences = list(product(range(grammaire.taille), repeat=grammaire.longueur))
    r_grad = np.array([grammaire.recompense_graduee(s) for s in sequences])
    r_spar = np.array([grammaire.recompense_tout_ou_rien(s) for s in sequences])
    valide = np.array([grammaire.analyser(s)["valide"] for s in sequences])

    betas = [0.01, 0.02, 0.05, 0.08, 0.12, 0.2, 0.35, 0.5]
    atteint = {0.01: 99.84, 0.02: 99.99, 0.05: 92.65, 0.08: 94.87,
               0.12: 57.13, 0.2: 20.59, 0.35: 5.27, 0.5: 3.01}

    def optimum(recompenses, beta):
        logits = recompenses / beta
        logits -= logits.max()
        p = np.exp(logits)
        return 100 * float(p[valide].sum() / p.sum())

    x = range(len(betas))
    plt.figure(figsize=(10, 5.5))
    plt.plot(x, [optimum(r_grad, b) for b in betas], "o-", color=BLEU,
             label="optimum of the GRADED reward (π*)")
    plt.plot(x, [optimum(r_spar, b) for b in betas], "^-", color=VERT,
             label="optimum of the ALL-OR-NOTHING reward (π*)")
    plt.plot(x, [atteint[b] for b in betas], "s--", color=ROUGE,
             label="actually reached by REINFORCE (graded)")
    plt.xticks(list(x), [str(b) for b in betas])
    plt.xlabel("entropy coefficient β")
    plt.ylabel("grammatical mass (%)")
    plt.ylim(-5, 108)
    plt.legend(fontsize=9)
    plt.title("The shaping tax: the graded reward's own optimum is worse\n"
              "and past β=0.08 the learned policy beats the optimum it aims at")
    plt.annotate("mode collapse is conservative:\nmore grammatical than π* itself",
                 xy=(3, 94.9), xytext=(3.4, 70), fontsize=8,
                 arrowprops=dict(arrowstyle="->", alpha=0.6))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "06_test2_shaping_tax.png"))
    plt.close()


def fig_recuit():
    """Le recuit de beta contre les deux regimes constants."""
    chemin = os.path.join(R2, "parametrisation_recuit.json")
    if not os.path.exists(chemin):
        print("  (donnees de recuit absentes)")
        return
    with open(chemin, encoding="utf-8") as f:
        donnees = json.load(f)
    cles = [k for k in donnees if k.startswith("recuit_")]
    fig, (g, d) = plt.subplots(1, 2, figsize=(13, 5))
    couleurs = [BLEU, VERT]
    for k, cle in enumerate(cles):
        trace = donnees[cle]
        eps = [t["episode"] for t in trace]
        etiquette = "anneal " + cle.replace("recuit_", "").replace("_", " → ")
        g.plot(eps, [t["valide_pct"] for t in trace], "o-", color=couleurs[k],
               linewidth=1.6, label=etiquette)
        d.plot(eps, [t["modes"] for t in trace], "o-", color=couleurs[k],
               linewidth=1.6, label=etiquette)
    g.axhline(99.99, color=ROUGE, linestyle="--", alpha=0.7,
              label="β constant 0.02 (18.6 modes)")
    g.axhline(57.13, color=ORANGE, linestyle="--", alpha=0.7,
              label="β constant 0.12 (45.9 modes)")
    g.set_xlabel("episode")
    g.set_ylabel("grammatical mass (%)")
    g.set_ylim(-5, 108)
    g.legend(fontsize=8)
    g.set_title("Validity during annealing", fontsize=10)

    d.axhline(48, color="black", linestyle=":", alpha=0.6, label="48 = the optimum")
    d.axhline(45.35, color="grey", linestyle="--", alpha=0.7,
              label="45.35 = uniform over the 6 determiners")
    d.axhline(18.6, color=ROUGE, linestyle="--", alpha=0.7, label="β constant 0.02")
    d.set_xlabel("episode")
    d.set_ylabel("effective modes (out of 48)")
    d.set_ylim(0, 52)
    d.legend(fontsize=8)
    d.set_title("Diversity during annealing", fontsize=10)
    fig.suptitle("The fix: annealing β dominates both constant regimes at once "
                 "— 99.97 % valid AND 45.3 / 48 modes", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "07_test2_annealing.png"))
    plt.close()


def fig_portrait(grammaire):
    """Portrait de phase.

    Les trajectoires n'avaient pas ete sauvegardees, donc il faut reentrainer
    3 graines x 20 000 episodes — environ 5 minutes. On les met en cache dans un
    .npy pour que les regenerations suivantes soient instantanees, et on affiche
    la progression : sans elle, la fonction reste muette assez longtemps pour
    donner l'impression d'un plantage.
    """
    import time
    sequences = list(product(range(grammaire.taille), repeat=grammaire.longueur))
    valide = np.array([grammaire.analyser(s)["valide"] for s in sequences])

    cache = os.path.join(R2, "trajectoires_acp.npy")
    if os.path.exists(cache):
        trajectoires = list(np.load(cache))
        print("    trajectoires rechargees depuis le cache")
    else:
        trajectoires = []
        for graine in (0, 1, 2):
            debut = time.time()
            fixer_graine(graine)
            politique = PolitiqueGRU(grammaire.taille)
            optimiseur = torch.optim.Adam(politique.parameters(), lr=1e-3)
            historique = deque(maxlen=100)
            _, p0 = distribution_exacte(politique, grammaire)
            points = [p0.double().numpy()]
            for episode in range(1, 20001):
                actions, log_probs, entropies, _ = politique.generer(
                    grammaire.longueur, taille_lot=1)
                r = grammaire.recompense_graduee(actions[0].tolist())
                baseline = sum(historique) / len(historique) if historique else 0.0
                historique.append(r)
                perte = -(log_probs.sum() * torch.tensor(r - baseline, dtype=torch.float32)) \
                        - 0.02 * entropies.sum()
                optimiseur.zero_grad()
                perte.backward()
                torch.nn.utils.clip_grad_norm_(politique.parameters(), 5.0)
                optimiseur.step()
                if episode % 250 == 0:
                    _, p = distribution_exacte(politique, grammaire)
                    points.append(p.double().numpy())
                if episode % 2500 == 0:
                    print(f"    seed {graine} : {episode}/20000 "
                          f"({time.time() - debut:.0f}s)", flush=True)
            trajectoires.append(np.array(points))
            print(f"    seed {graine} done in {time.time() - debut:.0f}s", flush=True)
        np.save(cache, np.array(trajectoires))
        print(f"    trajectoires mises en cache dans {cache}")

    toutes = np.concatenate(trajectoires, axis=0)
    centre = toutes.mean(axis=0)
    U, S, Vt = np.linalg.svd(toutes - centre, full_matrices=False)
    ratio = (S ** 2) / (S ** 2).sum()
    ideal = np.zeros(len(sequences))
    ideal[valide] = 1.0 / valide.sum()
    proj_ideal = (ideal - centre) @ Vt[:2].T

    # L'etoile est dessinee en premier et en dessous : les ronds de depart
    # tombent juste a cote et etaient masques par elle.
    plt.figure(figsize=(9, 7))
    plt.scatter(*proj_ideal, color="gold", marker="*", s=520, edgecolor="black",
                zorder=2, label="uniform over the 48 (the optimum)")
    couleurs = [BLEU, ROUGE, VERT]
    for k, pts in enumerate(trajectoires):
        proj = (pts - centre) @ Vt[:2].T
        plt.plot(proj[:, 0], proj[:, 1], "-", color=couleurs[k], alpha=0.8,
                 linewidth=1.3, label=f"seed {k}")
        plt.scatter(proj[0, 0], proj[0, 1], color=couleurs[k], marker="o", s=110,
                    edgecolor="white", linewidth=1.6, zorder=5)
        plt.scatter(proj[-1, 0], proj[-1, 1], color=couleurs[k], marker="s", s=110,
                    edgecolor="black", linewidth=1.2, zorder=4)
    distances = [float(np.linalg.norm((pts - centre) @ Vt[:2].T - proj_ideal, axis=1)[0])
                 for pts in trajectoires]
    finales = [float(np.linalg.norm((pts - centre) @ Vt[:2].T - proj_ideal, axis=1)[-1])
               for pts in trajectoires]
    print(f"    distance to optimum in this plane — start: "
          f"{np.mean(distances):.3f}, end: {np.mean(finales):.3f}")
    plt.xlabel(f"PC1 ({100*ratio[0]:.1f} % of the movement)")
    plt.ylabel(f"PC2 ({100*ratio[1]:.1f} %)")
    plt.title("Training trajectories in exact distribution space\n"
              "circle = start, square = end. Caveat: this plane holds only "
              f"{100*(ratio[0]+ratio[1]):.1f} % of the movement")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "08_test2_phase_portrait.png"))
    plt.close()


if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)
    courte = Grammaire(longue=False)
    for nom, fonction in [
            ("01 reward shapes", fig_test1_recompenses),
            ("02 heatmap", fig_test1_heatmap),
            ("03 transfer", fig_test1_transfert),
            ("04 grammars", fig_test2_grammaires),
            ("05 tradeoff", lambda: fig_compromis(courte)),
            ("06 shaping tax", lambda: fig_gibbs(courte)),
            ("07 annealing", fig_recuit),
            ("08 phase portrait", lambda: fig_portrait(courte))]:
        print(f"  {nom} ...")
        try:
            fonction()
        except Exception as e:
            print(f"    echec : {type(e).__name__}: {e}")
    print(f"\nFigures ecrites dans {FIGURES}")
