"""Tour 66 : observation post hoc. Le signe de D_d (mediane - noeud) change entre x petit (negatif) et
x grand (positif). Temps de relaxation du mode mou en pas d'Adam : tau = sqrt(v_e3) / (2 lr |lambda_mou|)
(z3 bouge de 2 lr m / sqrt(v_e3) ; lambda_mou du jacobien, verifier_tour66_forme_normale.py, va comme
sqrt(x) : -2,32e-11 a x = 1e-13, -7,29e-9 a 1e-8). Lignes recopiees des sorties du 13/10/2026
(verifier_tour66_paire_pli.py et, pour les deux dernieres, verifier_tour64_gain_et_r.py au tour 65).
Question : le signe de D_d se range-t-il sur tau ?
"""

import math

LAM_1E13 = 2.32e-11  # |lambda_mou| a x = 1e-13 ; on prend |lambda| = LAM_1E13 * sqrt(x / 1e-13)
LAM_1E8 = 7.29e-9    # valeur calculee a 1e-8, pour controler la loi en sqrt(x)

# lr, eps, x, mediane sqrt(v_e3), D_d (noeud)
LIGNES = [
    (0.05, 1e-8, 1e-13, 4.3719e-8, -8.5965e-6),
    (0.05, 1e-9, 1e-11, 4.6508e-8, -6.8112e-6),
    (0.02, 1.6e-9, 1.6e-14, 1.8164e-8, -5.4745e-6),
    (0.02, 1.6e-10, 1.6e-12, 1.9011e-8, -5.7507e-6),
    (0.02, 8e-9, 1.6e-9, 1.6755e-8, -7.4344e-7),
    (0.02, 1.6e-8, 1.6e-9, 1.5392e-8, +4.9036e-7),
    (0.05, 5e-8, 1e-8, 3.6770e-8, +2.0649e-6),
    (0.05, 1e-7, 1e-8, 2.9128e-8, +3.6178e-6),
    (0.02, 2e-8, 1e-8, 1.4713e-8, +2.8112e-7),
    (0.02, 2e-8, 4e-9, 1.4734e-8, +5.0789e-7),
]

if __name__ == "__main__":
    print(f"controle de la loi en sqrt(x) : LAM_1E13 * sqrt(1e5) = {LAM_1E13 * math.sqrt(1e5):.3e} contre {LAM_1E8:.3e} calcule a 1e-8")
    print("lr    eps      x         sqrt(v_e3)   |lambda_mou|   tau (pas)   D_d (noeud)")
    for lr, eps, x, sv, D in sorted(LIGNES, key=lambda t: t[3] / (2 * t[0] * LAM_1E13 * math.sqrt(t[2] / 1e-13)), reverse=True):
        lam = LAM_1E13 * math.sqrt(x / 1e-13)
        tau = sv / (2 * lr * lam)
        print(f"{lr:4.2f}  {eps:7.1e}  {x:8.1e}  {sv:.4e}   {lam:.3e}      {tau:9.0f}   {D:+.3e}")
