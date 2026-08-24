"""RDTRL — ce que 29 hypotheses mortes disent de la visee d'un audit.

dipankarsarkar propose deux lectures de « quatre audits d'affilee ont trouve
autre chose que leur cible » : (1) la cible avait deja recu de l'attention, donc
seule la nouveaute de la region compte ; (2) un audit trouve dans sa TRAVERSEE,
et la cible est un point dedans, donc la largeur explique tout. Il tranche pour
la deuxieme sur quatre points de cette semaine.

Le carnet en offre vingt-neuf, sur vingt-six jours. Chaque entree de §1 nomme la
mesure qui l'a tuee, donc chacune se code sur trois variables :

  vise    : la mesure qui a produit la preuve refutante a-t-elle ete lancee POUR
            tester cette proposition-la, au moment ou elle a ete lancee ;
  objet   : la proposition morte porte-t-elle sur le MONDE (le modele, la tache,
            le paysage) ou sur l'INSTRUMENT (un fichier, un defaut d'argument,
            une colonne, une quantite imprimee) ;
  sur_place : la preuve refutante existait-elle deja sur le disque avant que la
            question soit posee.

LIMITE, ecrite avant les chiffres et non apres. Ce registre ne contient que des
morts. Il n'enregistre pas les verifications qui n'ont rien trouve, donc il n'a
pas de denominateur et ne peut pas mesurer un RENDEMENT — ce qui est exactement
ce que sa lecture deux affirme. Il mesure la COMPOSITION des trouvailles, ce qui
est une autre question, et la seule que mes donnees repondent. Deuxieme limite :
§1 s'intitule « hypotheses que J'AI formulees », donc la colonne « qui a trouve »
est biaisee par la structure du registre lui-meme et n'est pas rapportee.
"""

from scipy import stats

RUPTURE = "14/08"

# (numero, date de mort, vise, objet, preuve deja sur le disque, ce qui a tue)
ENTREES = [
    ("1.1", "29/07", True, "monde", False, "3 graines de plus au meme beta"),
    ("1.2", "29/07", True, "monde", False, "sonde de capacite supervisee"),
    ("1.3", "29/07", True, "monde", False, "mesure directe de la prediction"),
    ("1.4", "29/07", True, "monde", False, "optimum de Gibbs + dominance"),
    ("1.5", "29/07", False, "monde", False, "le test 2 produit un contre-exemple"),
    ("1.6", "31/07", True, "monde", False, "70 graines a condition unique"),
    ("1.7", "31/07", False, "monde", True, "la table de saturation existante"),
    ("1.8", "31/07", True, "monde", False, "optimum_produit.py"),
    ("1.9", "11/08", True, "instrument", False, "le supremum de la nulle vaut 1"),
    ("1.10", "11/08", False, "monde", False, "l'echelle par transpositions"),
    ("1.11", "11/08", True, "monde", False, "la premisse du certificat"),
    ("1.12", "11/08", True, "monde", False, "le hessien au point de babil"),
    ("1.13", "11/08", False, "monde", True, "§6.5 mesurait deja les collisions"),
    ("1.14", "11/08", True, "monde", False, "gradient_premier_pas.py"),
    ("1.15", "11/08", True, "monde", False, "une preuve en une ligne"),
    ("1.16", "11/08", True, "monde", False, "z sur six valeurs d'epsilon"),
    ("1.17", "12/08", True, "monde", False, "correlation sur 210 runs"),
    ("1.18", "14/08", True, "instrument", False, "il pointe la ligne beta"),
    ("1.19", "15/08", False, "instrument", True, "le fichier de resultats jamais ouvert"),
    ("1.20", "15/08", True, "instrument", False, "sa nulle a effet plante"),
    ("1.21", "15/08", False, "instrument", True, "plancher et t dans le meme fichier"),
    ("1.22", "15/08", False, "instrument", True, "min, max, zeros exacts"),
    ("1.23", "16/08", False, "instrument", True, "loi_nulle_longue depuis le 11/08"),
    ("1.24", "17/08", False, "instrument", True, "le plafond de reservoir dans la source"),
    ("1.25", "17/08", False, "instrument", True, "n_restarts=24 au site d'appel"),
    ("1.26", "17/08", True, "instrument", False, "hypothese sur moi, verifiee"),
    ("1.27", "18/08", True, "instrument", False, "une graine independante"),
    ("1.28", "18/08", False, "monde", True, "la bijectivite des 1200 runs"),
    ("1.29", "19/08", True, "monde", False, "la cellule appariee a 20 000 pas"),
]

# Les quatre de son tableau, plus les deux de mon cote de ce tour, codes pareil.
SEMAINE = [
    ("atomes", "21/08", False, "instrument", True, "un manifeste"),
    ("phrase vs artefact", "21/08", False, "instrument", True, "une phrase, une table"),
    ("tolerance de boucle", "22/08", False, "instrument", True, "600 montees x 3276"),
    ("fidelite REINFORCE", "19/08", False, "instrument", False, "un balayage 2D"),
    ("plafond_beta", "21/08", False, "instrument", True, "||grad J|| a cote, 4 ordres lache"),
    ("47,3 % du voisinage", "22/08", False, "instrument", True, "echantillon=1200 en source"),
]


def jour(date):
    j, m = date.split("/")
    return int(m) * 31 + int(j)


def fisher(a, b, c, d):
    return stats.fisher_exact([[a, b], [c, d]])[1]


def bloc(nom, entrees):
    n = len(entrees)
    vise = sum(e[2] for e in entrees)
    instrument = sum(e[3] == "instrument" for e in entrees)
    sur_place = sum(e[4] for e in entrees)
    print(f"  {nom:<28}{n:>4}{vise:>10} ({vise / n:>5.0%}){instrument:>10}"
          f" ({instrument / n:>5.0%}){sur_place:>10} ({sur_place / n:>5.0%})")
    return n, vise, instrument, sur_place


if __name__ == "__main__":
    print("=== SA LECTURE DEUX, CONTRE 29 POINTS AU LIEU DE 4 ===")
    print("  Codage integral dans la source, une justification par ligne,")
    print("  pour qu'il puisse recoder et voir ce que le codage porte.\n")
    print(f"  {'periode':<28}{'n':>4}{'vise':>18}{'instrument':>19}"
          f"{'deja sur disque':>24}")

    avant = [e for e in ENTREES if jour(e[1]) < jour(RUPTURE)]
    apres = [e for e in ENTREES if jour(e[1]) >= jour(RUPTURE)]
    n_t, v_t, i_t, s_t = bloc("tout le carnet", ENTREES)
    n_a, v_a, i_a, s_a = bloc(f"avant le {RUPTURE}", avant)
    n_b, v_b, i_b, s_b = bloc(f"a partir du {RUPTURE}", apres)
    bloc("cette semaine (ses 4 + 2)", SEMAINE)

    print(f"\n  visee, avant contre apres : {v_a}/{n_a} contre {v_b}/{n_b}, "
          f"Fisher exact p = {fisher(v_a, n_a - v_a, v_b, n_b - v_b):.4f}")
    print(f"  instrument, avant contre apres : {i_a}/{n_a} contre {i_b}/{n_b}, "
          f"Fisher exact p = {fisher(i_a, n_a - i_a, i_b, n_b - i_b):.2e}")
    print(f"  deja sur disque, avant contre apres : {s_a}/{n_a} contre {s_b}/{n_b}, "
          f"Fisher exact p = {fisher(s_a, n_a - s_a, s_b, n_b - s_b):.4f}")

    print("\n  visee CONDITIONNELLE a l'objet, tout le carnet :")
    for objet in ("monde", "instrument"):
        lot = [e for e in ENTREES if e[3] == objet]
        v = sum(e[2] for e in lot)
        print(f"    {objet:<12}{v:>3} / {len(lot):<3}  visees  ({v / len(lot):.0%})")
    monde = [e for e in ENTREES if e[3] == "monde"]
    instr = [e for e in ENTREES if e[3] == "instrument"]
    print(f"    Fisher exact p = "
          f"{fisher(sum(e[2] for e in monde), len(monde) - sum(e[2] for e in monde), sum(e[2] for e in instr), len(instr) - sum(e[2] for e in instr)):.4f}")

    print("\n  et le croisement qui decide entre nos deux lectures :")
    for objet in ("monde", "instrument"):
        lot = [e for e in ENTREES if e[3] == objet]
        s = sum(e[4] for e in lot)
        print(f"    {objet:<12}preuve deja sur le disque : {s:>3} / {len(lot)}")

    print("\n  les entrees ou la preuve dormait sur le disque :")
    for e in ENTREES:
        if e[4]:
            print(f"    §{e[0]:<6}{e[1]}  {e[5]}")
