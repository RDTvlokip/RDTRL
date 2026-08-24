"""RDTRL — un audit qui a un denominateur, et ce qu'il trouve.

Sa question et la mienne portent sur le RENDEMENT d'une verification. Ni son
tableau de quatre lignes ni mes vingt-neuf entrees de carnet ne peuvent y
repondre : ce sont des registres de TROUVAILLES, donc des echantillons selectionnes
sur l'issue. Le defaut que cet echange passe vingt tours a nommer dans la science,
applique a nos deux epistemologies.

Un rendement demande une population definie avant de regarder. En voici une, la
plus petite qui soit exhaustible : **les arguments par defaut numeriques du depot
qui peuvent borner une recherche, un budget ou une affirmation**. Vingt-deux
sites, enumeres mecaniquement par l'AST, resolus un par un, negatifs compris.

Deuxieme population, decouverte en resolvant la premiere : **les fichiers ou UN
generateur unique traverse plusieurs calculs publies**. La graine y nomme un flux,
pas un etat, donc l'artefact ne suffit pas a refaire le nombre. Neuf fichiers.
Demonstration mesuree sur celui que j'ai su reproduire au bit pres.
"""

import numpy as np

from bornes_par_messages_distincts import chercher
from loi_nulle_longue import matrices_information_generale, statistiques

PUBLIE = {27: 0.052615549471, 26: 0.136208288684, 25: 0.178347460157,
          24: 0.185005350060, 23: 0.215204662069}

# (site, classe, verdict)
SITES = [
    ("rl_copie.entrainer(max_episodes=30000)", "borne", "NON RESOLU"),
    ("rl_copie.entrainer(seuil_convergence=0.99)", "definition", "sans objet"),
    ("test4_controle.cible_sans_recouvrement(graine=7)", "graine", "sans objet"),
    ("localisation_effondrement.entrainer_position_figee(graine=0)", "graine", "sans objet"),
    ("rl_grammaire.entrainer(max_episodes=20000)", "borne", "NON RESOLU"),
    ("appariement_vs_distance.concordance(n_paires=2000000)", "borne", "sain, SE ~ 3,5e-4 jamais imprimee"),
    ("appariement_vs_distance.recherche_pire_cas(n_restarts=24)", "borne", "DEFAUT, §1.25"),
    ("appariement_vs_distance.recherche_pire_cas(n_pas=60)", "borne", "sain : 0/24 epuises, longueur max 14"),
    ("bornes_par_messages_distincts.monter(pas=120)", "borne", "sain : 0 epuise"),
    ("bornes_par_messages_distincts (restarts=20)", "borne", "DEFAUT, trouve ici : +15,5 % a R=27"),
    ("certificat_deux_agents.monter(pas=4000)", "borne", "resolu par le nom de l'artefact"),
    ("certificat_deux_agents.bissection(seuil=0.5)", "definition", "sans objet"),
    ("certificat_deux_agents.est_loi_produit(tolerance=1e-09)", "borne", "sain : marge 3,6e+06"),
    ("correction_de_selection.route_1_vectorisee(bloc=20000)", "decoupage", "sans objet"),
    ("grammaire3.loi_nulle(graine=0)", "graine", "sans objet"),
    ("iso_echantillons.reinforce_variante(graine=0)", "graine", "sans objet"),
    ("loi_nulle_longue.tirer(reservoir=2000000)", "borne", "DEFAUT, §1.24, corrige"),
    ("plancher_de_detection.rupture(seuil=1.98)", "definition", "sans objet"),
    ("plancher_de_detection.rupture(kmax=40)", "borne", "sain : valeurs publiees 2 a 4"),
    ("queue_de_inflation.tirer(graine=0)", "graine", "sans objet"),
    ("realisabilite_treillis.monter(pas=300)", "borne", "sain : 0/1800 epuises"),
    ("representable_atteignable_stable.reinforce(graine=0)", "graine", "sans objet"),
    ("tools.bench_device.mesurer(n_episodes=200)", "banc", "sans objet"),
]

FILES_PARTAGES = [
    ("appariement_vs_distance.py", 19),
    ("representable_atteignable_stable.py", 19),
    ("gradient_premier_pas.py", 15),
    ("courbe_de_contrainte.py", 12),
    ("code_emergent.py", 9),
    ("bornes_par_messages_distincts.py", 8),
    ("dynamique_uniforme.py", 7),
    ("qui_ecrit_le_code.py", 5),
    ("effet_par_beta.py", 4),
]


def ecart(lot):
    cm, ca, _ = statistiques(matrices_information_generale(lot))
    return cm - ca


if __name__ == "__main__":
    print("=== POPULATION 1 : LES 23 DEFAUTS NUMERIQUES QUI PEUVENT BORNER ===")
    print("  Enumeres par l'AST sur tout le depot, pas choisis. Negatifs compris.\n")
    bornes = [s for s in SITES if s[1] == "borne"]
    for site, classe, verdict in SITES:
        marque = " <-" if verdict.startswith("DEFAUT") else ""
        print(f"  {site:<60}{verdict}{marque}")
    defauts = [s for s in bornes if s[2].startswith("DEFAUT")]
    inconnus = [s for s in bornes if s[2] == "NON RESOLU"]
    print(f"\n  bornes reelles {len(bornes)} / {len(SITES)} sites")
    print(f"  defauts {len(defauts)}   sains {len(bornes) - len(defauts) - len(inconnus)}"
          f"   non resolus {len(inconnus)}")
    print(f"  rendement sur les bornes resolues : "
          f"{len(defauts)} / {len(bornes) - len(inconnus)} = "
          f"{len(defauts) / (len(bornes) - len(inconnus)):.0%}")

    print("\n=== POPULATION 2 : LA GRAINE NOMME UN FLUX, PAS UN ETAT ===")
    print("  `bornes_par_messages_distincts_g0.json` publie graine 0, restarts 20,")
    print("  pas 120. Ces trois nombres ne determinent PAS ses cinq bornes : le")
    print("  generateur est file a travers les cinq planchers, donc l'ORDRE compte,")
    print("  et l'ordre n'est nulle part dans l'artefact.\n")

    print(f"  {'ordre':>26}{'plancher':>10}{'borne':>18}{'publie':>18}{'ecart':>14}")
    for nom, ordre in (("celui du script (27->23)", [27, 26, 25, 24, 23]),
                       ("inverse (23->27)", [23, 24, 25, 26, 27])):
        generateur = np.random.default_rng(0)
        for plancher in ordre:
            v, _ = chercher(ecart, plancher, generateur, 20, 120)
            print(f"  {nom:>26}{plancher:>10}{v:>18.12f}{PUBLIE[plancher]:>18.12f}"
                  f"{v - PUBLIE[plancher]:>+14.9f}")
        print()

    print("  fichiers ou un generateur unique traverse >= 4 calculs publies :")
    for f, n in FILES_PARTAGES:
        print(f"    {f:<46}{n:>4} consommations")
    print(f"\n  {len(FILES_PARTAGES)} fichiers. Correctif : un generateur par calcul")
    print("  publie, ou la position dans le flux imprimee a cote du nombre.")
