"""Figure de synthese du 31/07/2026 : quatre facons de comparer les memes runs.

Quatre panneaux, quatre questions, toutes tranchees dans la meme journee :

  A  les deux chemins numeriques, graine par graine. Le coin tient, le
     remplissage non.
  B  le plafond de produit n'est jamais franchi, et les modes effectifs tombent
     sur des produits d'entiers.
  C  gradient exact contre echantillonne : ce qui les separe est la PROFONDEUR
     de l'effondrement transitoire, pas le point d'arrivee.
  D  l'arret precoce ne gagne rien, sauf dans le coin au plafond le plus haut.

Conventions de couleur : deux teintes categorielles seulement, validees en mode
"toutes paires" (CVD dE 24,7 ; vision normale 33,6). Le texte reste en encre et
ne porte jamais la couleur d'une serie ; la legende est presente des qu'il y a
deux series.
"""

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R2 = os.path.join(RACINE, "results_test2")
ARCHIVE = os.path.join(RACINE, "results_test2_float32")
SORTIE = os.path.join(RACINE, "figures")

SURFACE = "#fcfcfb"
ENCRE = "#0b0b0b"
ENCRE_2 = "#52514e"
MUET = "#8a8983"
BLEU = "#2a78d6"
ORANGE = "#eb6834"
PLAFOND = {"sg": 12.0, "pl": 24.0}


def charger(*candidats):
    for c in candidats:
        if os.path.exists(c):
            with open(c, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(candidats[0])


def style(ax, titre, sous_titre=None, chemin=None):
    """chemin : quel chemin numerique a produit ces donnees.

    Non facultatif en pratique : trois panneaux sur quatre viennent d'un chemin
    different, et on vient justement de montrer que le chemin deplace les
    chiffres. Une figure qui ne le dit pas invite a comparer ce qui ne se
    compare pas.
    """
    ax.set_facecolor(SURFACE)
    ax.set_title(titre, color=ENCRE, fontsize=11.5, fontweight="bold",
                 loc="left", pad=16 if sous_titre else 8)
    if sous_titre:
        ax.text(0, 1.02, sous_titre, transform=ax.transAxes, color=ENCRE_2,
                fontsize=9, va="bottom")
    if chemin:
        ax.text(1.0, 1.02, chemin, transform=ax.transAxes, color=MUET,
                fontsize=8.5, va="bottom", ha="right")
    ax.grid(True, color=MUET, alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    for c in ("top", "right"):
        ax.spines[c].set_visible(False)
    for c in ("left", "bottom"):
        ax.spines[c].set_color(MUET)
        ax.spines[c].set_linewidth(0.8)
    ax.tick_params(colors=ENCRE_2, labelsize=9)


def panneau_a(ax):
    """Les deux chemins, graine par graine."""
    a = {l["graine"]: l for l in charger(
        os.path.join(ARCHIVE, "balayage_70_graines_b0.02_fusion.json"),
        os.path.join(R2, "balayage_70_graines_b0.02_fusion.json"))["detail"]}
    b = {l["graine"]: l for l in charger(
        os.path.join(R2, "balayage_70_graines_b0.02_float64_fusion.json"))["detail"]}
    g = sorted(set(a) & set(b))

    lim = 26
    ax.plot([0, lim], [0, lim], color=MUET, linewidth=1.2, linestyle=(0, (4, 3)),
            zorder=1)
    ax.text(24.4, 25.2, "identical", color=MUET, fontsize=8.5, ha="right")
    for coin, couleur in (("sg", BLEU), ("pl", ORANGE)):
        ks = [k for k in g if a[k]["branche"] == coin]
        ax.scatter([a[k]["modes_effectifs"] for k in ks],
                   [b[k]["modes_effectifs"] for k in ks],
                   s=64, color=couleur, alpha=0.72, linewidths=1.4,
                   edgecolors=SURFACE, zorder=3,
                   label=f"{'singular' if coin=='sg' else 'plural'} corner "
                         f"(n={len(ks)})")
        p = PLAFOND[coin]
        for f in (ax.axvline, ax.axhline):
            f(p, color=couleur, linewidth=1.1, linestyle=(0, (2, 3)),
              alpha=0.55, zorder=2)
    identiques = sum(1 for k in g
                     if abs(a[k]["modes_effectifs"] - b[k]["modes_effectifs"]) < 0.05)
    meme_coin = sum(1 for k in g if a[k]["branche"] == b[k]["branche"])
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("effective modes, float32 path", color=ENCRE_2, fontsize=9.5)
    ax.set_ylabel("effective modes, float64 path", color=ENCRE_2, fontsize=9.5)
    style(ax, "A · Same start, different trajectory",
          f"{meme_coin}/70 keep their corner · only {identiques}/70 keep their mode count",
          chemin="both paths")
    # En bas a droite : en haut a gauche la boite chevauchait la ligne de
    # plafond pluriel a 24.
    ax.legend(frameon=False, fontsize=8.5, loc="lower right",
              labelcolor=ENCRE_2, handletextpad=0.5)


def panneau_b(ax):
    """Le plafond n'est jamais franchi, et les modes sont des produits."""
    # Chemin float64, celui qui est canonique depuis le 31/07/2026. Le panneau A
    # montre que le choix ne change ni le coin ni le respect du plafond.
    d = charger(os.path.join(R2, "balayage_70_graines_b0.02_float64_fusion.json"))["detail"]
    rng = np.random.default_rng(0)
    for i, (coin, couleur) in enumerate((("pl", ORANGE), ("sg", BLEU))):
        v = np.array([l["modes_effectifs"] for l in d if l["branche"] == coin])
        y = i + rng.uniform(-0.17, 0.17, len(v))
        ax.scatter(v, y, s=52, color=couleur, alpha=0.7, linewidths=1.3,
                   edgecolors=SURFACE, zorder=3,
                   label=f"{'plural' if coin=='pl' else 'singular'} corner "
                         f"(n={len(v)})")
        p = PLAFOND[coin]
        ax.plot([p, p], [i - 0.34, i + 0.34], color=couleur, linewidth=2.4,
                solid_capstyle="round", zorder=4)
        ax.text(p + 0.5, i + 0.30, f"largest valid product = {p:.0f}",
                color=ENCRE_2, fontsize=8.5, va="center")
        ax.text(p + 0.5, i + 0.06, f"{(np.abs(v - p) < 0.05).sum()} runs sit exactly on it",
                color=MUET, fontsize=8, va="center")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["plural", "singular"])
    ax.set_xlim(0, 30); ax.set_ylim(-0.6, 1.75)
    ax.set_xlabel("effective modes at convergence", color=ENCRE_2, fontsize=9.5)
    style(ax, "B · The ceiling is never crossed",
          "0 violations in 70 runs · values land on integer products",
          chemin="float64 path")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left",
              labelcolor=ENCRE_2, handletextpad=0.5)


def panneau_c(ax):
    """Exact contre echantillonne : la profondeur de l'effondrement."""
    r = charger(os.path.join(R2, "trajectoire_couplage_b0.02.json"))
    for proc, couleur, nom in (("exact", BLEU, "exact gradient"),
                               ("echantillonne", ORANGE, "sampled REINFORCE")):
        premier = True
        for t in [x for x in r["trajectoires"] if x["procedure"] == proc]:
            h = t["historique"]
            pas = np.array([max(e["pas"], 1) for e in h])
            mo = np.array([e["modes_effectifs"] for e in h])
            ax.plot(pas, mo, color=couleur, linewidth=2.0, alpha=0.85, zorder=3,
                    label=nom if premier else None)
            k = int(np.argmin(mo))
            ax.scatter([pas[k]], [mo[k]], s=90, color=couleur, zorder=5,
                       linewidths=1.6, edgecolors=SURFACE)
            premier = False
    ax.set_xscale("log")
    ax.set_ylim(-2, 54)
    # Les deux procedures n'ont pas la meme unite : un pas exact traite les
    # 8 000 sequences, un pas echantillonne en tire une. On le dit plutot que de
    # laisser croire a une abscisse commune.
    ax.set_xlabel("step — full-batch for exact, one episode for sampled (log)",
                  color=ENCRE_2, fontsize=9.5)
    ax.set_ylabel("effective modes", color=ENCRE_2, fontsize=9.5)
    ax.axhline(47.5, color=MUET, linewidth=1, linestyle=(0, (2, 3)), zorder=1)
    ax.text(1.15, 49.4, "untrained network: 47.5 modes", color=MUET, fontsize=8.5)
    ax.annotate("crushed to a single sentence", xy=(450, 1.3), xytext=(1.9, 15),
                color=ENCRE_2, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=MUET, linewidth=1))
    style(ax, "C · What separates the two procedures is the collapse, not the end",
          "dots mark each run's minimum · exact stays above 10.7, sampled reaches 1.09",
          chemin="float64 path")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right",
              labelcolor=ENCRE_2, handletextpad=0.5)


def panneau_d(ax):
    """L'arret precoce, 20 graines, sur le chemin float64.

    Le titre a refuter — "+12,5 modes en s'arretant tot" — est ne au §7.9, dans
    stabilite_et_trajectoire.py, dont la ligne 79 est deja la version float64.
    Sa valeur finale de 11,5 modes le confirme : le float32 donne 18,6. C'est
    donc sur CE chemin qu'il faut le mesurer, et c'est aussi le canonique.
    """
    # L'etiquette suit la donnee et n'est jamais ecrite en dur : sinon un repli
    # silencieux affiche le nom d'un chemin en montrant les chiffres de l'autre.
    lignes, chemin_utilise = [], "float64 path"
    for f in sorted(glob.glob(os.path.join(R2, "chemin_avantage_float64_arrondi_*.json"))):
        lignes += charger(f).get("partie_b", [])
    if not lignes:
        chemin_utilise = "float32 path — float64 not measured yet"
        for f in sorted(glob.glob(os.path.join(R2, "chemin_avantage_float32_*.json"))):
            lignes += charger(f).get("partie_b", [])
    donnees = []
    for l in lignes:
        h = [e for e in l["historique"] if e["masse_valide_pct"] >= 90.0]
        if not h:
            continue
        mo = [e["modes_effectifs"] for e in h]
        donnees.append((max(mo), l["modes_fin"], l["branche"]))
    donnees.sort(key=lambda x: x[0] - x[1])
    ecarts = np.array([p - f for p, f, _ in donnees])

    for i, (pic, fin, coin) in enumerate(donnees):
        couleur = BLEU if coin == "sg" else ORANGE
        # Seuil a 0,5 mode : en dessous, l'ecart n'a aucune portee pour la
        # question posee, et superposer les deux marques donne un croissant.
        if pic - fin < 0.5:
            # Pic et fin confondus : un seul point plein. Superposer les deux
            # marques donnait un croissant illisible sur 17 des 20 lignes.
            ax.scatter([fin], [i], s=52, color=couleur, edgecolors=SURFACE,
                       linewidths=1.2, zorder=4)
            continue
        ax.plot([fin, pic], [i, i], color=couleur, linewidth=2.2, alpha=0.55,
                solid_capstyle="round", zorder=2)
        ax.scatter([fin], [i], s=48, color=SURFACE, edgecolors=couleur,
                   linewidths=1.8, zorder=3)
        ax.scatter([pic], [i], s=52, color=couleur, edgecolors=SURFACE,
                   linewidths=1.2, zorder=4)
    for coin, couleur, nom in (("sg", BLEU, "singular corner"),
                               ("pl", ORANGE, "plural corner")):
        n = sum(1 for *_, c in donnees if c == coin)
        ax.plot([], [], color=couleur, linewidth=2.2, label=f"{nom} (n={n})")
    ax.set_xlabel("effective modes — single dot: no gap · open dot final, filled dot peak",
                  color=ENCRE_2, fontsize=9.5)
    ax.set_ylabel("20 seeds, sorted by gap", color=ENCRE_2, fontsize=9.5)
    ax.set_yticks([])
    ax.set_xlim(0, 27)
    # Le detail par coin est CALCULE : la version precedente disait "all three
    # plural" en dur, ce qui etait vrai du float32 et faux du float64, ou trois
    # des cinq sont singuliers.
    coins = np.array([c for *_, c in donnees])
    gros = ecarts > 1
    par_coin = " · ".join(f"{(gros & (coins == c)).sum()} {n}"
                          for c, n in (("pl", "plural"), ("sg", "singular")))
    style(ax, "D · Early stopping gains nothing in most runs",
          f"median gap {np.median(ecarts):+.2f} · {gros.sum()}/{len(ecarts)} above one mode "
          f"({par_coin})", chemin=chemin_utilise)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right",
              labelcolor=ENCRE_2, handletextpad=0.5)


def main():
    os.makedirs(SORTIE, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), facecolor=SURFACE)
    for f, ax in zip((panneau_a, panneau_b, panneau_c, panneau_d), axes.ravel()):
        f(ax)
    fig.suptitle("RDTRL — test 2, four comparisons on the same runs",
                 color=ENCRE, fontsize=14, fontweight="bold", x=0.008, ha="left",
                 y=0.985)
    fig.text(0.008, 0.955,
             "70 seeds at beta = 0.02 unless stated. Every number is exact: the "
             "8,000-sequence space is enumerated, not sampled.",
             color=ENCRE_2, fontsize=9.5, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    chemin = os.path.join(SORTIE, "comparaison_test2.png")
    fig.savefig(chemin, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"Ecrit dans {chemin}")


if __name__ == "__main__":
    main()
