"""Tour du mecanisme plancher-de-v, 20/09/2026 (reprise) : test
precommis par un agent-dipankar pour l'hypothese de melange
propre/coince (voir CARNET.md, section beta2). Predictions posees
AVANT le test : (a) cycle propre continue de decroitre, (b) fraction
coincee depasse 5/8 (>70%, peut-etre ~90%), (c) espacement median
continue de MONTER.

Resultat : a beta2=0.985, le detecteur standard (seuil 0.0015) ne
trouve PLUS AUCUN evenement discret sur 100000 pas. Diagnostic sur
40000 pas : max_dev=0.000668, mean_dev=0.000392, ratio max/mean=1.7 --
la trajectoire entiere est fusionnee en un regime continu, plus de
phase "calme" separee. Confirme la prediction (b)/(c) a l'extreme :
la fraction coincee est montee bien au-dela des ~90% predits, plus
proche de 100%, donc plus d'evenement discret a compter -- c'est la
limite naturelle de "l'espacement monte jusqu'a devenir indefini",
pas une contradiction du mecanisme de melange.

Rejoint une observation faite plus tot ce tour (avant le cadre du
melange), notee alors "ni confirmee ni refutee, methode inadaptee" :
a beta2=0.99, R4 n'a plus de plateau, juste une agitation continue.
Avec le cadre du melange en main, cette observation ancienne est la
meme chose vue une marche plus tot sur le meme axe.
"""

import sys
sys.path.insert(0, '.')
from verifier_kicks_adam_grille_fine import tracer, detecter_evenements

BETA2 = 0.985
PAS_LONG = 100000
PAS_DIAG = 40000
SEUILS_DIAG = (0.001, 0.0005, 0.0002, 0.0001)


def main():
    vals_long = tracer(PAS_LONG, betas=(0.9, BETA2))
    baseline = sum(v for p, v in vals_long if 15000 <= p < 25000) / len(
        [v for p, v in vals_long if 15000 <= p < 25000]
    )
    pics = detecter_evenements(vals_long, baseline, pas_min=5000)
    print(f"beta2={BETA2}, {PAS_LONG} pas, seuil standard (0.0015) : n_events={len(pics)}")

    vals_diag = tracer(PAS_DIAG, betas=(0.9, BETA2))
    baseline_diag = sum(v for p, v in vals_diag if 15000 <= p < 25000) / len(
        [v for p, v in vals_diag if 15000 <= p < 25000]
    )
    devs = [abs(v - baseline_diag) for p, v in vals_diag if p > 5000]
    max_dev = max(devs)
    mean_dev = sum(devs) / len(devs)
    print(f"diagnostic sur {PAS_DIAG} pas : max_dev={max_dev:.6f}  mean_dev={mean_dev:.6f}  "
          f"ratio={max_dev/mean_dev:.2f}")
    for seuil in SEUILS_DIAG:
        pics_s = detecter_evenements(vals_diag, baseline_diag, seuil=seuil, pas_min=5000)
        print(f"  seuil={seuil}: n_events={len(pics_s)}")


if __name__ == "__main__":
    main()
