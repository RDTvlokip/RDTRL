"""Le test de renversement en deux panneaux.

A  la grammaire a trois genres : les modes effectifs des 70 runs, coin par coin,
   avec le plafond calcule AVANT le lancement. Jamais franchi.
B  le test quantitatif, celui qu'un renommage ne peut pas passer : le rapport
   des plafonds change de 2 a 3, et le rapport des moyennes observees suit.

Memes conventions que figure_comparaison.py : deux teintes categorielles
validees en mode toutes paires, texte en encre, plafonds en trait plein.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R2 = os.path.join(RACINE, "results_test2")
SORTIE = os.path.join(RACINE, "figures")

SURFACE, ENCRE, ENCRE_2, MUET = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
BLEU, ORANGE = "#2a78d6", "#eb6834"

JEUX = {
    "standard, 2 genders": ("balayage_70_graines_b0.02_float64_fusion.json",
                            {"sg": 12.0, "pl": 24.0}),
    "three genders":       ("balayage_70_graines_b0.02_float64_trois_genres_fusion.json",
                            {"sg": 36.0, "pl": 12.0}),
}


def charger(nom):
    with open(os.path.join(R2, nom), encoding="utf-8") as f:
        return json.load(f)["detail"]


def style(ax, titre, sous_titre=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(titre, color=ENCRE, fontsize=11.5, fontweight="bold", loc="left",
                 pad=16 if sous_titre else 8)
    if sous_titre:
        ax.text(0, 1.02, sous_titre, transform=ax.transAxes, color=ENCRE_2,
                fontsize=9, va="bottom")
    ax.grid(True, color=MUET, alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    for c in ("top", "right"):
        ax.spines[c].set_visible(False)
    for c in ("left", "bottom"):
        ax.spines[c].set_color(MUET)
        ax.spines[c].set_linewidth(0.8)
    ax.tick_params(colors=ENCRE_2, labelsize=9)


def panneau_a(ax):
    """Trois genres : les 70 runs contre leur plafond."""
    fichier, plaf = JEUX["three genders"]
    d = charger(fichier)
    rng = np.random.default_rng(0)
    for i, (coin, couleur, nom) in enumerate(((("pl"), ORANGE, "plural corner"),
                                              (("sg"), BLEU, "singular corner"))):
        v = np.array([l["modes_effectifs"] for l in d if l["branche"] == coin])
        ax.scatter(v, i + rng.uniform(-0.17, 0.17, len(v)), s=54, color=couleur,
                   alpha=0.7, linewidths=1.3, edgecolors=SURFACE, zorder=3,
                   label=f"{nom} (n={len(v)})")
        p = plaf[coin]
        ax.plot([p, p], [i - 0.34, i + 0.34], color=couleur, linewidth=2.6,
                solid_capstyle="round", zorder=4)
        ax.text(p + 0.7, i + 0.29, f"ceiling = {p:.0f}, predicted before the run",
                color=ENCRE_2, fontsize=8.5, va="center")
        ax.text(p + 0.7, i + 0.05,
                f"max observed {v.max():.1f} · {int((np.abs(v-p)<0.05).sum())} runs exactly on it",
                color=MUET, fontsize=8, va="center")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["plural", "singular"])
    ax.set_xlim(0, 44); ax.set_ylim(-0.6, 1.75)
    ax.set_xlabel("effective modes at convergence", color=ENCRE_2, fontsize=9.5)
    style(ax, "A · Three genders: the ceilings swap AND change value",
          "70 seeds, β = 0.02 · zero violations in either corner")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=ENCRE_2,
              handletextpad=0.5)


def panneau_b(ax):
    """Le test qu'un renommage ne peut pas passer : le rapport."""
    noms, rapports_p, rapports_o = [], [], []
    for nom, (fichier, plaf) in JEUX.items():
        d = charger(fichier)
        moy = {c: np.mean([l["modes_effectifs"] for l in d if l["branche"] == c])
               for c in ("sg", "pl")}
        haut = max(plaf, key=plaf.get); bas = min(plaf, key=plaf.get)
        noms.append(nom)
        rapports_p.append(plaf[haut] / plaf[bas])
        rapports_o.append(moy[haut] / moy[bas])

    x = np.arange(len(noms))
    ax.plot([-0.5, len(noms) - 0.5], [1, 1], color=MUET, linewidth=1,
            linestyle=(0, (2, 3)), zorder=1)
    for i, (p, o) in enumerate(zip(rapports_p, rapports_o)):
        ax.plot([i, i], [p, o], color=MUET, linewidth=1.4, alpha=0.6, zorder=2)
        ax.scatter([i], [p], s=150, marker="_", color=ENCRE, linewidths=3, zorder=4)
        ax.scatter([i], [o], s=110, color=ORANGE if i else BLEU, zorder=5,
                   linewidths=1.4, edgecolors=SURFACE)
        # Quand predit et observe sont quasi confondus (3,0 contre 3,01), les
        # deux etiquettes se superposent : on les ecarte verticalement.
        proche = abs(p - o) < 0.15
        ax.text(i + 0.09, p + (0.11 if proche else 0), f"predicted {p:.1f}",
                color=ENCRE_2, fontsize=9, va="center")
        ax.text(i + 0.09, o - (0.11 if proche else 0), f"observed {o:.2f}",
                color=ENCRE_2, fontsize=9, va="center")
    ax.set_xticks(x); ax.set_xticklabels(noms)
    ax.set_xlim(-0.5, len(noms) + 0.12); ax.set_ylim(0.6, 3.6)
    ax.set_ylabel("ratio of the two corners", color=ENCRE_2, fontsize=9.5)
    style(ax, "B · A relabelling cannot change a ratio",
          "dash = ceiling ratio, computed · dot = ratio of observed means")
    ax.text(0.5, 0.72, "my first reversal test only swapped the labels,\n"
                       "so it could only ever land on the left column",
            color=MUET, fontsize=8.5, ha="center")


def main():
    os.makedirs(SORTIE, exist_ok=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 6.0), facecolor=SURFACE,
                               gridspec_kw={"width_ratios": [1.45, 1]})
    panneau_a(a); panneau_b(b)
    fig.suptitle("RDTRL — the reversal test: is the ceiling a law or my lexicon?",
                 color=ENCRE, fontsize=14, fontweight="bold", x=0.008, ha="left",
                 y=0.99)
    fig.text(0.008, 0.936,
             "Both grammars have equal-sized corners. Only the largest valid "
             "product differs, and it is an isomorphism invariant.",
             color=ENCRE_2, fontsize=9.5, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    chemin = os.path.join(SORTIE, "renversement_test2.png")
    fig.savefig(chemin, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"Ecrit dans {chemin}")


if __name__ == "__main__":
    main()
