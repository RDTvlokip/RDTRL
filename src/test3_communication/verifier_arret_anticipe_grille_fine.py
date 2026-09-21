"""Question 13 des 20 (ETAT.md), correction du 21/09/2026 : la grille
de phase de `verifier_arret_anticipe_vs_phase_kick.py`
(`DEBUTS_TESTES`, pas de 40) etait trop grossiere -- meme famille de
biais que la question 3 (`pas_min=5000`). Trouve par un agent d'audit
(worktree isole), verifie independamment ici, pas=1 sur les 1000 pas
de depart possibles.

Deux corrections par rapport a la premiere publication :
1. Le RATIO max/min n'est pas monotone a grille fine a eps=1e-9
   (2,010 -> 1,925 -> 1,953) -- ma conclusion "contre-intuitive,
   croit avec la patience" etait un artefact de sous-echantillonnage.
   Ce qui survit : `extra := duree_max - patience` croit bien de
   facon monotone (202 -> 370 -> 572), signal plus faible.
2. eps=3e-9 cache une VRAIE transition discontinue de `extra` (73 a
   patience<=460, saute a 465 exactement a patience=465, saute encore
   a 529 exactement a patience=529) -- deux sauts nets, pas une pente.
   Confirme par le test precommis de l'agent (balayage fin de
   patience sur [400,600], resolution 1) : la transition bascule en
   UN SEUL pas de patience, deux fois -- pas progressive.

Coincidence notable, non expliquee analytiquement : extra=465 apres
le premier saut est quasi identique a la periode du cycle de kick
(~456-500 pas, verifier_kicks_adam_grille_fine.py).

n_floor (fraction des departs de phase qui tombent exactement sur
duree=patience) est quasi invariant a la patience -- la vraie
"vulnerabilite" est un effet de queue (8% a 33-36% des phases selon
eps), pas une fonction lisse de la patience comme le ratio le
suggerait.
"""

import sys
sys.path.insert(0, '.')

from verifier_arret_anticipe_vs_phase_kick import tracer, simuler_arret, EPS_AMELIORATION_VALEURS, PATIENCES

PAS_DEBUT_MIN = 9700
PAS_DEBUT_MAX = 10700


def balayage_fin(vals, eps, patience):
    durees = []
    n_floor = 0
    for debut in range(PAS_DEBUT_MIN, PAS_DEBUT_MAX, 1):
        pas_arret = simuler_arret(vals, debut, patience, eps)
        if pas_arret is not None:
            d = pas_arret - debut
            durees.append(d)
            if d == patience:
                n_floor += 1
    return durees, n_floor


def localiser_transition(vals, eps, patience_min, patience_max):
    resultats = []
    for patience in range(patience_min, patience_max + 1):
        durees, _ = balayage_fin(vals, eps, patience)
        extra_max = max(durees) - patience if durees else None
        resultats.append((patience, extra_max))
    return resultats


def main():
    vals = tracer()

    print("=== Comparaison grille grossiere (pas=40, deja publiee) vs fine (pas=1) ===")
    for eps in EPS_AMELIORATION_VALEURS:
        for patience in PATIENCES:
            durees, n_floor = balayage_fin(vals, eps, patience)
            ratio = max(durees) / min(durees)
            extra_max = max(durees) - patience
            print(f"eps={eps}  patience={patience}  ratio={ratio:.3f}  "
                  f"extra_max={extra_max}  n_floor={n_floor}/{len(durees)}")

    print("\n=== Localisation fine de la transition a eps=3e-9, patience in [400,600] ===")
    for patience, extra_max in localiser_transition(vals, 3e-9, 400, 600):
        print(f"patience={patience}  extra_max={extra_max}")


if __name__ == "__main__":
    main()
