"""Le coefficient 0,2212 cite dans docs/REPONSE_ORDRE53.md ("the slowdown
coefficient 0.2212*sqrt(delta_c-delta)") n'existait nulle part ailleurs
dans le depot -- signale par l'agent-dipankar le 17/09/2026 ("a number
that got written down without being computed"). Il vient bien d'un calcul
reel (les trois points de residu publies par dipankar au tour 51,
verifies independamment dans cette session), mais jamais sauve en script.
Corrige ici : les trois points, le calcul du coefficient, et une mise en
garde sur ce qu'il mesure reellement (le residu de la loi molle en
UNITES s3, sur des points Adam, pas SGD -- donc pas touche par
l'artefact de gel du referent 3 sous SGD trouve le meme jour).
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
