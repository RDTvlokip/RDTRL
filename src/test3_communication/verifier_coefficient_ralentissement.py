"""Le coefficient 0,2212 cite dans docs/REPONSE_ORDRE53.md ("the slowdown
coefficient 0.2212*sqrt(delta_c-delta)") n'existait nulle part ailleurs
dans le depot -- signale par l'agent-dipankar le 17/09/2026 ("a number
that got written down without being computed"). Il vient bien d'un calcul
reel, mais jamais sauve en script -- corrige ici avec les trois points et
le calcul du coefficient.

CORRECTION du 17/09/2026 (verifier_gap_racines.py) : la remarque
precedente de ce docstring ("mesure sur des trajectoires ADAM") etait
FAUSSE. Ce residu n'est PAS un ecart mesure par entrainement -- c'est
l'ecart ALGEBRIQUE entre la racine stable et la racine instable du
systeme couple (d3_instable - d3_stable), qui se calcule par recherche
de racines pure, sans entrainer quoi que ce soit. Verifie : recalcule de
ce gap_d3 algebrique aux 3 memes deltas donne un ratio publie/algebrique
de 1,0000 / 0,9999 / 1,0000. Etendu a 11 points dans
verifier_gap_racines.py : le coefficient converge vers 0,221305-0,221372
tout pres de delta_c (confirmation du 0,2212), et vaut 0,278 loin du pli
(la "derive" est confirmee et affinee, pas un artefact de 3 points).
"""

import math

# (delta, distance_au_pli delta_c-delta, residu en s3 = R_mesure - R_predit_converti)
# publies par dipankarsarkar au tour 51 (REPONSE_ORDRE52.md), verifies
# independamment (voir REPONSE_ORDRE52.md, section "One thing your Adam
# hypothesis predicts...").
POINTS = [
    (0.0100000, 3.437e-03, 1.468e-02),
    (0.0130000, 4.372e-04, 4.700e-03),
    (0.0134300, 7.210e-06, 5.943e-04),
]

if __name__ == "__main__":
    print("=== coefficient du 'gap' branche-selle en unites s3, gap/sqrt(delta_c-delta) ===")
    coeffs = []
    for delta, gap_dc, gap_s3 in POINTS:
        c = gap_s3 / math.sqrt(gap_dc)
        coeffs.append(c)
        print(f"  delta={delta}  gap_s3={gap_s3:.4e}  delta_c-delta={gap_dc:.4e}  coefficient={c:.6f}")

    print(f"\n  moyenne des trois : {sum(coeffs)/len(coeffs):.4f}")
    print(f"  ecart entre le premier et le dernier point : {coeffs[0]-coeffs[-1]:.4f}"
          f"  ({(coeffs[0]-coeffs[-1])/coeffs[-1]*100:.1f} % de derive)")
    print("\n  MISE EN GARDE : ce coefficient decrit le residu de la loi molle en s3,")
    print("  mesure sur des trajectoires ADAM (pas SGD) -- il n'est donc pas directement")
    print("  touche par l'artefact 'referent 3 gele sous SGD' trouve le 17/09/2026.")
    print("  Mais il n'a jamais ete verifie que ces trois points Adam eux-memes")
    print("  entrainaient reellement le referent 3 de facon comparable -- a verifier.")
